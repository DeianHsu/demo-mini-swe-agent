import json
from textwrap import dedent

from mini_swe_agent_demo.docker_bash.use_llm import use_llm_with_tools
from mini_swe_agent_demo.docker_bash.environment import execute, limit_observation

MAX_TOOL_HISTORY_CHARS = 256000
OMITTED_OBSERVATION = "[更早的命令输出已省略；需要时请缩小范围重新读取。]"


def trim_tool_history(history: list[dict]) -> None:
    tools = [row for row in history if row.get("role") == "tool" and isinstance(row.get("content"), str)]
    for row in tools:
        row["content"] = limit_observation(row["content"])
    total = sum(len(row["content"]) for row in tools)
    for row in tools:
        if total <= MAX_TOOL_HISTORY_CHARS:
            break
        if len(row["content"]) > len(OMITTED_OBSERVATION):
            total -= len(row["content"]) - len(OMITTED_OBSERVATION)
            row["content"] = OMITTED_OBSERVATION


def model(task: str, history: list, workdir: str = "/workspace") -> dict:
    trim_tool_history(history)
    messages = []
    prompt = dedent(f"""
        你是一个编程智能体。
        执行环境是 Linux 容器中的 Bash，当前目录为 {workdir}。
        需要操作时，通过 execute 工具提交 Bash 脚本。
        每次调用启动独立进程，文件修改会保留。
        任务完成后，以普通文本回复 finish。
    """).strip()
    messages.append({
        "role": "system",
        "content": prompt
    })
    messages.append({
        "role": "user",
        "content": task
    })
    messages.extend(history)
    response = use_llm_with_tools(messages)
    return response


def agent(task: str, *, container_name: str = "mini-swe-learning", workdir: str = "/workspace", timeout: int = 10,
          max_steps: int = 10, ) -> list:
    history = []
    for _ in range(max_steps):
        assistant_message = model(task, history, workdir=workdir)
        history.append(assistant_message)
        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            print("模型正文：", assistant_message.get("content"))
            break
        else:
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]

                if tool_name != "execute":
                    result = f"Error: 未知工具 {tool_name}"
                else:
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError) as e:
                        result = f"Error: 参数解析失败：{e}"
                    else:
                        if not isinstance(arguments, dict):
                            result = "Error: 参数必须是 JSON 对象"
                        elif not isinstance(arguments.get("command"), str):
                            result = "Error: command 必须是字符串"
                        else:
                            result = execute(arguments["command"],
                                             container_name=container_name,
                                             workdir=workdir,
                                             timeout=timeout, )
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result
                })

    return history


if __name__ == "__main__":
    task = dedent(r"""
        查看当前工作目录，列出其中的文件，读取 hello.txt 的完整内容。
        只执行查看操作，完成后回复 finish。
    """).strip()
    history = agent(task)
    for msg in history:
        print(msg)
