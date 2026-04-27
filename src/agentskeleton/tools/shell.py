import subprocess
import time
from typing import Any, ClassVar

from agentskeleton.tools.base import Tool, ToolContext, ToolResult


def _truncate(text: str | None, max_bytes: int) -> tuple[str, bool]:
    if text is None:
        text = ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


class ShellTool(Tool):
    name: ClassVar[str] = "shell"
    description: ClassVar[str] = "Run a shell command in the configured workspace."
    risk: ClassVar[str] = "shell"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run in the workspace.",
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        command = str(args["command"])
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=context.workspace,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=context.shell_timeout_seconds,
                check=False,
            )
            duration = time.perf_counter() - started
            stdout, stdout_truncated = _truncate(
                completed.stdout,
                context.shell_max_output_bytes,
            )
            stderr, stderr_truncated = _truncate(
                completed.stderr,
                context.shell_max_output_bytes,
            )
            payload = {
                "command": command,
                "working_directory": str(context.workspace),
                "exit_code": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "duration_seconds": duration,
                "timed_out": False,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
            return ToolResult(
                success=completed.returncode == 0,
                payload=payload,
                summary=f"Command exited with {completed.returncode}",
                error=None if completed.returncode == 0 else "Command failed",
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - started
            stdout, stdout_truncated = _truncate(
                exc.stdout or "",
                context.shell_max_output_bytes,
            )
            stderr, stderr_truncated = _truncate(
                exc.stderr or "",
                context.shell_max_output_bytes,
            )
            return ToolResult(
                success=False,
                payload={
                    "command": command,
                    "working_directory": str(context.workspace),
                    "exit_code": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "duration_seconds": duration,
                    "timed_out": True,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
                summary=(
                    f"Command timed out after "
                    f"{context.shell_timeout_seconds} seconds"
                ),
                error="Command timed out",
            )
