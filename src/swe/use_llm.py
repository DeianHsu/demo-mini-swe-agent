# 配置模型连接，并将原生函数调用的响应拆分为对话消息和运行元数据。
import os

from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 加载模型连接配置，供下面的客户端读取。
load_dotenv()

# 复用同一个客户端，并设置请求超时。
# 关闭软件开发工具包（SDK）内部重试，使每轮请求可以明确记账。
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    timeout=120.0,
    max_retries=0,
)

# 向模型发送对话与工具定义，分别返回回复消息和包含用量的响应元数据。
def use_llm_with_tools(messages: list[dict]) -> tuple[dict, dict]:
    # 当前价目只覆盖 deepseek-flash，防止换用其他模型后仍按旧价计费。
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    if model_name != "deepseek-flash":
        raise ValueError(
            "当前预算价目只适用于 deepseek-flash；更换模型前须更新价目"
        )

    # 声明 execute 工具及其必需的 command 参数，让模型按约定提交 Bash 脚本。
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "execute",
                "description": "在容器工作区执行 Bash 脚本",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "需要执行的 Bash 脚本"
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    ]

    # 将本轮消息和工具定义发送给模型，并明确关闭额外的思考模式。
    response = client.chat.completions.create(
        tools=tools_schema,
        model=model_name,
        messages=messages,
        extra_body={
            "thinking": {"type": "disabled"}
        },
    )

    # 对话消息供后续轮次使用，响应标识、停止原因和用量供轨迹与成本统计使用。
    # 保留缺失的 usage 为 None，让调用方明确识别无法核算的响应。
    choice = response.choices[0]
    return choice.message.to_dict(), {
        "response_id": response.id,
        "model": response.model,
        "created": response.created,
        "finish_reason": choice.finish_reason,
        "usage": response.usage.to_dict() if response.usage is not None else None,
    }
