import subprocess
import time
from typing import Any, ClassVar

from agentskeleton.tools.base import Tool, ToolContext, ToolResult

MAX_COMMAND_BYTES = 16_384
UNINSPECTABLE_VALUE = "<uninspectable>"


def _exception_text(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:
        return type(exc).__name__


def _utf8_size(value: str) -> int | None:
    try:
        return len(value.encode("utf-8"))
    except Exception:
        return None


def _safe_strip(value: str) -> str | None:
    try:
        stripped = value.strip()
    except Exception:
        return None
    return str.__str__(stripped)


def _truncate(text: str | bytes | None, max_bytes: int) -> tuple[str, bool]:
    if text is None:
        text = ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    try:
        encoded = text.encode("utf-8")
    except Exception:
        return UNINSPECTABLE_VALUE, False
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
        raw_command = args.get("command")
        if not isinstance(raw_command, str):
            return ToolResult(
                success=False,
                summary="Command invalid: command must be a string",
                error="Command invalid",
            )
        stripped_command = _safe_strip(raw_command)
        if stripped_command is None:
            return ToolResult(
                success=False,
                summary="Command invalid: command could not be inspected",
                error="Command invalid",
            )
        if not stripped_command:
            return ToolResult(
                success=False,
                summary="Command invalid: command cannot be blank",
                error="Command invalid",
            )
        command_bytes = _utf8_size(raw_command)
        if command_bytes is None:
            return ToolResult(
                success=False,
                summary="Command invalid: command could not be inspected",
                error="Command invalid",
            )
        if command_bytes > MAX_COMMAND_BYTES:
            return ToolResult(
                success=False,
                summary="Command invalid: command too large",
                error="Command invalid",
            )
        command = str.__str__(raw_command)
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
        except OSError as exc:
            duration = time.perf_counter() - started
            stderr, stderr_truncated = _truncate(
                _exception_text(exc),
                context.shell_max_output_bytes,
            )
            return ToolResult(
                success=False,
                payload={
                    "command": command,
                    "working_directory": str(context.workspace),
                    "exit_code": None,
                    "stdout": "",
                    "stderr": stderr,
                    "duration_seconds": duration,
                    "timed_out": False,
                    "stdout_truncated": False,
                    "stderr_truncated": stderr_truncated,
                },
                summary=f"Command launch failed: {type(exc).__name__}",
                error="Command launch failed",
            )
