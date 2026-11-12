"""离线验证原生函数调用循环和工具历史限额。"""

import os
import unittest
from unittest.mock import call, patch

with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-test-key"}), patch("dotenv.load_dotenv"):
    from mini_swe_agent_demo.docker_bash import mini_agent


def execute_call(call_id: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "execute", "arguments": arguments},
    }


class MiniAgentTests(unittest.TestCase):
    def test_model_bounds_tool_history_without_losing_call_pairs(self):
        history = []
        for i in range(30):
            history.extend([
                {"role": "assistant", "tool_calls": [execute_call(str(i), '{"command":"pwd"}')]},
                {"role": "tool", "tool_call_id": str(i), "content": str(i) + "x" * 15990},
            ])
        final = {"role": "assistant", "content": "finish"}
        with patch.object(mini_agent, "use_llm_with_tools", return_value=final) as request:
            mini_agent.model("检查文件", history, workdir="/app")

        tools = history[1::2]
        self.assertLessEqual(sum(len(row["content"]) for row in tools), mini_agent.MAX_TOOL_HISTORY_CHARS)
        self.assertEqual([row["tool_call_id"] for row in tools], [str(i) for i in range(30)])
        self.assertEqual([row["tool_calls"][0]["id"] for row in history[::2]], [str(i) for i in range(30)])
        self.assertEqual(tools[0]["content"], mini_agent.OMITTED_OBSERVATION)
        self.assertTrue(tools[-1]["content"].startswith("29"))
        self.assertEqual(len(history), 60)
        messages = request.call_args.args[0]
        self.assertIn("/app", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "检查文件"})
        self.assertEqual(messages[2:], history)

    def test_tool_results_are_returned_to_the_model_with_matching_ids(self):
        action = {"role": "assistant", "tool_calls": [
            execute_call("call-1", '{"command":"pwd"}'),
            execute_call("call-2", '{"command":"cat hello.txt"}'),
        ]}
        final = {"role": "assistant", "content": "finish"}
        with (
            patch.object(mini_agent, "use_llm_with_tools", side_effect=[action, final]) as request,
            patch.object(mini_agent, "execute", side_effect=["/app", "Hello Agent!"]) as execute,
            patch("builtins.print"),
        ):
            history = mini_agent.agent("读取文件", container_name="offline", workdir="/app", timeout=30)

        self.assertEqual(execute.call_args_list, [
            call("pwd", container_name="offline", workdir="/app", timeout=30),
            call("cat hello.txt", container_name="offline", workdir="/app", timeout=30),
        ])
        self.assertEqual(history, [
            action,
            {"role": "tool", "tool_call_id": "call-1", "content": "/app"},
            {"role": "tool", "tool_call_id": "call-2", "content": "Hello Agent!"},
            final,
        ])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[1].args[0][2:], history[:-1])

    def test_invalid_arguments_are_reported_without_executing_commands(self):
        cases = [
            ("{", "参数解析失败"),
            ("[]", "参数必须是 JSON 对象"),
            ('{"command":1}', "command 必须是字符串"),
        ]
        for arguments, error in cases:
            with self.subTest(arguments=arguments):
                action = {"role": "assistant", "tool_calls": [execute_call("call-1", arguments)]}
                final = {"role": "assistant", "content": "finish"}
                with (
                    patch.object(mini_agent, "use_llm_with_tools", side_effect=[action, final]),
                    patch.object(mini_agent, "execute") as execute,
                    patch("builtins.print"),
                ):
                    history = mini_agent.agent("检查文件")
                execute.assert_not_called()
                self.assertEqual(history[1]["tool_call_id"], "call-1")
                self.assertIn(error, history[1]["content"])

    def test_max_steps_stops_repeated_tool_calls(self):
        actions = [
            {"role": "assistant", "tool_calls": [execute_call(str(i), '{"command":"pwd"}')]}
            for i in range(2)
        ]
        with (
            patch.object(mini_agent, "use_llm_with_tools", side_effect=actions) as request,
            patch.object(mini_agent, "execute", return_value="/workspace") as execute,
        ):
            history = mini_agent.agent("检查文件", max_steps=2)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
