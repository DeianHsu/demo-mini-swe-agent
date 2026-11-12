"""离线验证 Docker Bash 命令执行、输出限制及镜像工具路径。"""

import subprocess
import unittest
from unittest.mock import patch

from mini_swe_agent_demo.docker_bash import environment


class DockerEnvironmentTests(unittest.TestCase):
    def test_output_limit_keeps_short_output_and_both_ends_of_large_output(self):
        short = "Exit code:0\nSTDOUT:\n正常输出\nSTDERR:\n"
        self.assertEqual(environment.limit_observation(short), short)
        long = "start\n" + "中文🙂" * 100000 + "\nlast error"
        limited = environment.limit_observation(long)
        self.assertLessEqual(len(limited), environment.MAX_OBSERVATION_CHARS)
        self.assertTrue(limited.startswith("start\n"))
        self.assertTrue(limited.endswith("last error"))
        self.assertIn("输出已截断", limited)

    def test_execute_limits_large_output_and_keeps_exit_code_and_error(self):
        result = subprocess.CompletedProcess([], 1, "first line\n" + "x" * 8_000_000, "real compile error")
        with patch.object(environment.subprocess, "run", return_value=result):
            observation = environment.execute("cat huge.log", container_name="offline", workdir="/app")
        self.assertLessEqual(len(observation), environment.MAX_OBSERVATION_CHARS)
        self.assertTrue(observation.startswith("Exit code:1\n"))
        self.assertIn("first line", observation)
        self.assertIn("STDERR:\nreal compile error", observation)
        self.assertIn("输出已截断", observation)

    def test_go_command_uses_the_image_environment(self):
        def simulated_shell(*, args, **kwargs):
            if "-lc" in args:
                return subprocess.CompletedProcess(args, 127, "", "bash: go: command not found")
            return subprocess.CompletedProcess(args, 0, "go version go1.24.3 linux/amd64", "")

        with patch.object(environment.subprocess, "run", side_effect=simulated_shell):
            observation = environment.execute("go version", container_name="offline", workdir="/app")
        self.assertIn("Exit code:0", observation)
        self.assertIn("go1.24.3", observation)

if __name__ == "__main__":
    unittest.main()
