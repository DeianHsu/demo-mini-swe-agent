import subprocess

MAX_OBSERVATION_CHARS = 16000


def limit_observation(text: str) -> str:
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    notice = f"\n[输出已截断；原始长度 {len(text)} 字符。请缩小命令范围继续读取。]\n"
    available = MAX_OBSERVATION_CHARS - len(notice)
    head = available // 2
    return text[:head] + notice + text[-(available - head):]


def execute(command: str, *, container_name: str = "mini-swe-learning", workdir: str = "/workspace",
            timeout: int = 10, ) -> str:
    args = [
        "docker",
        "exec",
        "-w",
        workdir,
        container_name,
        "bash",
        "-c",
        command,
    ]

    try:
        result = subprocess.run(
            args=args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
        )
        output = ""
        output += f"Exit code:{result.returncode}\n"
        output += "STDOUT:\n" + result.stdout + "\n"
        output += "STDERR:\n" + result.stderr + "\n"
        return limit_observation(output)
    except subprocess.TimeoutExpired:
        return "Error: command timeout"
    except OSError as e:
        return f"Error: process failed: {e}"

