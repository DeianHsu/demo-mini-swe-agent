# mini-swe-agent-demo

一个参考 [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) 手写的 Python 智能体（Agent）学习项目。

模型通过原生函数调用（Function Calling）提交 Bash 命令，在 Docker 容器中读取文件、修改代码和运行测试。程序将执行结果写回消息历史，驱动下一轮操作。

## 功能

- 单一工具 `execute`：解析 `tool_calls` 中的 `command`，交给容器中的 `bash -c` 执行。
- 工具结果通过关联 `tool_call_id` 的 `tool` 消息反馈给模型。
- 模型停止工具调用后结束，默认最多调用模型 10 轮，单条命令等待 10 秒。
- 单次命令输出最多 16,000 字符，工具历史正文累计最多 256,000 字符；截断时保留调用对应关系。

## 项目结构

```text
src/mini_swe_agent_demo/
├── __init__.py
└── docker_bash/
    ├── __init__.py
    ├── mini_agent.py        # 消息历史、工具调度与智能体循环
    ├── environment.py       # Docker Bash 命令执行与输出截断
    └── use_llm.py           # 模型请求与原生函数调用定义
```

`model()` 组织提示与消息历史，`agent()` 控制循环并调度工具，`execute()` 执行命令并返回退出码、标准输出和标准错误。每次命令启动独立进程，容器内文件修改会保留，跨次调用的 `cd` 和环境变量设置不会保留。

## 安装与配置

需要 Python 3.14+、uv 和运行中的 Docker Linux 引擎。Windows 可使用 Docker Desktop。

在项目根目录安装依赖：

```powershell
uv sync
```

在根目录创建 `.env`，填写模型服务配置：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=模型接口地址
DEEPSEEK_MODEL=模型名称
```

模型接口通过 OpenAI Python 软件开发工具包（SDK）调用。当前请求使用 DeepSeek 的 `thinking` 参数，服务需要支持 `tools`。`.env` 已加入 Git 忽略规则。

## 运行示例

以下命令均在项目根目录的 PowerShell 中执行，运行时会调用配置的模型服务。

首次创建示例容器并准备文件：

```powershell
docker run -d --name mini-swe-learning -w /workspace python:3.14-slim sleep infinity
docker exec -w /workspace mini-swe-learning bash -c 'echo Hello Agent! > hello.txt'
```

容器运行后启动智能体：

```powershell
uv run python -m mini_swe_agent_demo.docker_bash.mini_agent
```

默认任务列出工作目录并读取 `hello.txt`。任务位于 `docker_bash/mini_agent.py` 的 `if __name__ == "__main__":` 中，可改为修复容器内项目并运行测试。容器名、工作目录、命令等待时间和模型轮数可通过 `agent()` 参数调整。

已有容器停止后可通过 `docker start mini-swe-learning` 再次启动。

## 测试

离线测试使用模拟的模型和 Docker 执行结果，覆盖工具调用与结果回填、输出限制、历史限额及容器工具路径：

```powershell
uv run python -m unittest discover -s tests -v
```

## 评测结果

SWE-bench Pro V2 共 642 题，累计通过 542 题、未通过 100 题，有效评分覆盖全部题目。累计通过占比为 **84.42%**，包含历史重试与补丁重评，不能作为单次通过率。评测配置、各仓库成绩和统计口径见[完整评测报告](docs/SWE-bench-Pro-V2-最终报告-2026-10-26.md)。
