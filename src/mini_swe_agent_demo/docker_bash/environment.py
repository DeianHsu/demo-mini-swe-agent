# 在已有容器中执行 Bash 命令，并将命令结果整理为模型可读取的观察文本。
import subprocess

# 限制返回给模型的单次命令输出长度，避免大量日志占满对话上下文。
MAX_OBSERVATION_CHARS = 16000


# 限制观察文本的长度，超出上限时保留开头和结尾，并提示模型缩小读取范围。
def limit_observation(text: str) -> str:
    # 未超限的输出直接返回，保留完整内容。
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    # 先为截断提示预留空间，再将剩余字符额度分配给原文的首尾。
    notice = f"\n[输出已截断；原始长度 {len(text)} 字符。请缩小命令范围继续读取。]\n"
    available = MAX_OBSERVATION_CHARS - len(notice)
    head = available // 2
    return text[:head] + notice + text[-(available - head):]


# 在指定容器和目录内执行一条命令，返回退出码、标准输出和标准错误。
def execute(command: str, *, container_name: str = "mini-swe-learning",
            workdir: str = "/workspace", timeout: int = 60) -> str:
    # 在容器内用 timeout 限制实际命令的运行时间，超时发送 KILL 信号。
    # Bash 使用非登录模式，沿用镜像原有的工具路径等环境配置。
    args = [
        "docker", "exec", "-w", workdir, container_name,
        "timeout", "--signal=KILL", "--verbose", str(timeout),
        "bash", "-c", command,
    ]

    # 收集命令输出，并替换无法按 UTF-8 解码的字节。
    # 宿主进程额外等待 15 秒，为容器命令结束及通信收尾留出时间。
    try:
        result = subprocess.run(
            args=args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 15,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        # 执行通道仍未返回时向上抛出异常，由预算监管流程结束本次任务。
        raise RuntimeError(
            "Docker 执行通道超时，本题应结束并停止容器"
        ) from error

    # 按固定格式拼接执行结果，再限制其长度后交给智能体。
    output = (
        f"Exit code:{result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )
    return limit_observation(output)
