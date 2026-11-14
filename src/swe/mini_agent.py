# 组织模型对话、执行工具调用并保存轨迹；整体运行时限由 budget 模块监管。
import json
from decimal import Decimal
from pathlib import Path
from textwrap import dedent

from swe.use_llm import use_llm_with_tools
from swe.environment import execute, limit_observation
from swe.cost import (
    TOKEN_PRICES,
    calculate_cost,
)


# 为历史工具输出设置总字符上限，并用占位文本替换过早的大段输出。
MAX_TOOL_HISTORY_CHARS = 256000
OMITTED_OBSERVATION = "[更早的命令输出已省略；需要时请缩小范围重新读取。]"


# 原地压缩历史工具输出，保留消息顺序与工具调用标识之间的对应关系。
def trim_tool_history(history: list[dict]) -> None:
    # 只处理含文本的工具消息，先限制每条观察结果的长度。
    tools = [row for row in history if row.get("role") == "tool" and isinstance(row.get("content"), str)]
    for row in tools:
        row["content"] = limit_observation(row["content"])
    # 若总量仍超限，从最早的工具输出开始替换，优先保留近期信息。
    total = sum(len(row["content"]) for row in tools)
    for row in tools:
        if total <= MAX_TOOL_HISTORY_CHARS:
            break
        # 只替换比占位文本更长的内容，避免压缩后反而增加字符数。
        if len(row["content"]) > len(OMITTED_OBSERVATION):
            total -= len(row["content"]) - len(OMITTED_OBSERVATION)
            row["content"] = OMITTED_OBSERVATION


# 为一轮模型请求组装系统提示、任务和历史消息，并返回回复及其元数据。
def model(task: str, history: list, workdir: str = "/workspace", *,
          thinking_mode: str = "disabled") -> tuple[dict, dict]:
    # 请求前先压缩工具历史，控制传给模型的上下文规模。
    trim_tool_history(history)
    # 系统提示约定工作目录、工具使用方式以及普通文本结束回复。
    messages = []
    prompt = dedent(f"""
        你是一个编程智能体。
        执行环境是 Linux 容器中的 Bash，当前目录为 {workdir}。
        需要操作时，通过 execute 工具提交 Bash 脚本。
        每次调用启动独立进程，文件修改会保留。
        任务完成后，以普通文本回复 finish。
    """).strip()
    # 按系统提示、任务描述、历史对话的顺序拼接消息。
    messages.append({
        "role": "system",
        "content": prompt
    })
    messages.append({
        "role": "user",
        "content": task
    })
    messages.extend(history)
    # 调用模型适配层，将本轮结果交回智能体循环处理。
    response = use_llm_with_tools(messages, thinking_mode=thinking_mode)
    return response

# 将运行事件按一行一条 JSON 的格式追加到轨迹文件，供后续恢复用量和排查问题。
def record_event(path: Path | None, kind: str, data: dict) -> None:
    # 未指定路径时不落盘；指定路径时每条事件写完即关闭文件，减少中断时的记录丢失。
    if path is not None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {"kind": kind, "data": data},
                    ensure_ascii=False,
                ) + "\n"
            )

# 循环请求模型并执行工具，直到模型结束或达到轮数、费用阈值。
# 整体任务时限由外层预算进程负责，此函数负责每轮对话和单条命令的执行。
def agent(task: str, *, container_name: str = "mini-swe-learning",
          workdir: str = "/workspace", timeout: int = 60,
          max_steps: int = 250, cost_limit: float = 3.0,
          trajectory_path: Path | None = None, thinking_mode: str = "disabled") -> dict:
    # 拒绝无效预算，避免开始执行后才发现轮数或时限不可用。
    if max_steps < 1 or timeout <= 0 or cost_limit <= 0:
        raise ValueError("轮数、命令时限和费用阈值必须大于零")

    # 初始化对话、累计费用和轮数；使用 Decimal 进行金额计算。
    # 默认结束原因是轮数用尽，其他结束路径会覆盖该状态。
    history = []
    total_cost = Decimal("0")
    cost_known = True
    limit = Decimal(str(cost_limit))
    steps = 0
    exit_reason = "step_limit"

    # 保存本次任务、预算及实际采用的价目，使后续统计沿用本次运行的计价口径。
    record_event(trajectory_path, "start", {
        "task": task,
        "workdir": workdir,
        "max_steps": max_steps,
        "command_timeout": timeout,
        "cost_limit_usd": str(limit),
        "thinking_mode": thinking_mode,
        "token_prices_per_million": {
            key: str(value) for key, value in TOKEN_PRICES.items()
        },
    })

    try:
        # 每次循环对应一轮模型请求，一轮回复可以包含多个工具调用。
        for _ in range(max_steps):
            # 在下一轮请求前检查费用；刚返回的一轮可能使累计费用超过阈值。
            if total_cost >= limit:
                exit_reason = "cost_limit"
                break

            # 先记录请求轮数，并将本轮费用标为未知，避免中断时把缺失消耗当作零。
            steps += 1
            cost_known = False
            record_event(trajectory_path, "request", {"step": steps})
            # 请求模型并把回复加入历史，供下一轮理解此前的操作。
            message, metadata = model(task, history, workdir=workdir, thinking_mode=thinking_mode)
            history.append(message)

            # 工具执行前先保存回复与用量，后续命令失败时仍可恢复这一轮的消耗。
            record_event(trajectory_path, "assistant", {
                "step": steps,
                "message": message,
                **metadata,
            })

            # 用量是核算费用的必需信息，缺失时终止任务并保留异常原因。
            usage = metadata["usage"]
            if usage is None:
                raise ValueError("模型响应缺少 usage，无法核算费用")

            # 累加本轮已确认的费用，并恢复费用已知标志。
            total_cost += calculate_cost(usage)
            cost_known = True

            # 无工具调用时按停止原因结束循环；是否真正解题成功由外部评测判断。
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                exit_reason = (
                    "finished"
                    if metadata["finish_reason"] == "stop"
                    else "model_incomplete"
                )
                break

            # 按模型给出的顺序处理本轮工具调用，每个调用都生成对应的工具回复。
            for tool_call in tool_calls:
                function = tool_call["function"]
                # 未声明的工具返回可读错误，让模型有机会在下一轮纠正请求。
                if function["name"] != "execute":
                    observation = f"Error: 未知工具 {function['name']}"
                else:
                    # 解析模型生成的 JSON 参数，将格式错误作为工具观察返回。
                    try:
                        arguments = json.loads(function["arguments"])
                    except (json.JSONDecodeError, TypeError) as error:
                        observation = f"Error: 参数解析失败：{error}"
                    else:
                        # 参数必须是对象，且 command 必须是字符串，才能交给执行层。
                        if not isinstance(arguments, dict):
                            observation = "Error: 参数必须是 JSON 对象"
                        elif not isinstance(arguments.get("command"), str):
                            observation = "Error: command 必须是字符串"
                        else:
                            # 在指定容器、工作目录和单命令时限内执行合法指令。
                            observation = execute(
                                arguments["command"],
                                container_name=container_name,
                                workdir=workdir,
                                timeout=timeout,
                            )

                # 用调用标识关联工具输出与请求，再同时加入对话历史和运行轨迹。
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": observation,
                }
                history.append(tool_message)
                record_event(trajectory_path, "tool", tool_message)

    except Exception as error:
        # 记录本层无法恢复的异常，再向调用方抛出，避免把执行失败伪装成正常结束。
        exit_reason = "error"
        record_event(trajectory_path, "error", {
            "type": type(error).__name__,
            "message": str(error),
        })
        raise

    finally:
        # 保存结束原因、轮数和已确认费用；最后一次请求的用量未知时，费用保持 None。
        summary = {
            "exit_reason": exit_reason,
            "model_calls": steps,
            "thinking_mode": thinking_mode,
            "budget_cost_usd": str(total_cost) if cost_known else None,
        }
        record_event(trajectory_path, "end", summary)

    # 将结束摘要与对话历史交给调用方，由调用方决定展示或保存哪些内容。
    return {**summary, "history": history}


# 直接运行此模块时，通过预算入口执行一个只读的容器演示任务。
if __name__ == "__main__":
    # 演示所需的监管入口仅在直接运行时加载，导入智能体函数时不加载它。
    from uuid import uuid4

    from swe.budget import run_case

    # 演示任务用于检查工作目录与文件读取是否可用。
    task = dedent(r"""
        查看当前工作目录，列出其中的文件，读取 hello.txt 的完整内容。
        只执行查看操作，完成后回复 finish。
    """).strip()

    # 为每次演示分配唯一输出目录，并启动带整体时限的解题进程。
    output = Path("archive/demo") / uuid4().hex
    result = run_case(
        task,
        "mini-swe-learning",
        "/workspace",
        output,
    )
    # 展示轨迹保存位置和最终摘要，方便查看本次运行结果。
    print(f"轨迹目录：{output.resolve()}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
