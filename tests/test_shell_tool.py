import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from agentskeleton.tools.base import ToolContext
from agentskeleton.tools.shell import ShellTool


def command_for(code: str) -> str:
    return subprocess.list2cmdline([sys.executable, "-c", code])


def test_shell_tool_captures_stdout(tmp_path: Path) -> None:
    result = ShellTool().execute(
        {"command": command_for("print('hello')")},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["exit_code"] == 0
    assert result.payload["stdout"].strip() == "hello"
    assert result.payload["stderr"] == ""


def test_shell_tool_returns_nonzero_exit_code(tmp_path: Path) -> None:
    result = ShellTool().execute(
        {
            "command": command_for(
                "import sys; print('bad', file=sys.stderr); sys.exit(2)"
            )
        },
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.payload["exit_code"] == 2
    assert result.payload["stderr"].strip() == "bad"


def test_shell_tool_times_out(tmp_path: Path) -> None:
    result = ShellTool().execute(
        {"command": command_for("import time; time.sleep(2)")},
        ToolContext(workspace=tmp_path, shell_timeout_seconds=1),
    )

    assert result.success is False
    assert result.payload["timed_out"] is True
    assert result.error == "Command timed out"


def test_shell_tool_truncates_output(tmp_path: Path) -> None:
    result = ShellTool().execute(
        {"command": command_for("print('abcdef')")},
        ToolContext(workspace=tmp_path, shell_max_output_bytes=4),
    )

    assert result.payload["stdout"] == "abcd"
    assert result.payload["stdout_truncated"] is True


def test_shell_tool_uses_utf8_replacement_for_output_decoding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_kwargs = {}

    def fake_run(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="한글", stderr="")

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute(
        {"command": "echo test"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert seen_kwargs["encoding"] == "utf-8"
    assert seen_kwargs["errors"] == "replace"


def test_shell_tool_handles_missing_captured_streams(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=None, stderr=None)

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute(
        {"command": "echo test"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["stdout"] == ""
    assert result.payload["stderr"] == ""
