import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from agentskeleton.tools.base import ToolContext
from agentskeleton.tools.shell import ShellTool


def command_for(code: str) -> str:
    return subprocess.list2cmdline([sys.executable, "-c", code])


class UnencodableString(str):
    def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("cannot encode")


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


def test_shell_tool_rejects_missing_command(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute({}, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.error == "Command invalid"
    assert result.summary == "Command invalid: command must be a string"
    assert result.payload == {}


def test_shell_tool_rejects_non_string_command(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute({"command": 123}, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.error == "Command invalid"
    assert result.summary == "Command invalid: command must be a string"
    assert result.payload == {}


def test_shell_tool_rejects_blank_command(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute({"command": "   "}, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.error == "Command invalid"
    assert result.summary == "Command invalid: command cannot be blank"
    assert result.payload == {}


def test_shell_tool_rejects_oversized_command(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute(
        {"command": "x" * 16_385},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Command invalid"
    assert result.summary == "Command invalid: command too large"
    assert result.payload == {}


def test_shell_tool_rejects_unencodable_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute(
        {"command": UnencodableString("echo test")},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Command invalid"
    assert result.summary == "Command invalid: command could not be inspected"
    assert result.payload == {}


def test_shell_tool_serializes_unencodable_process_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=UnencodableString("sk-secret123"),
            stderr=UnencodableString("stderr-secret"),
        )

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute(
        {"command": "echo test"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["stdout"] == "<uninspectable>"
    assert result.payload["stderr"] == "<uninspectable>"
    assert "sk-secret123" not in str(result.payload)
    assert "stderr-secret" not in str(result.payload)


def test_shell_tool_times_out(tmp_path: Path) -> None:
    result = ShellTool().execute(
        {"command": command_for("import time; time.sleep(2)")},
        ToolContext(workspace=tmp_path, shell_timeout_seconds=1),
    )

    assert result.success is False
    assert result.payload["timed_out"] is True
    assert result.error == "Command timed out"


def test_shell_tool_decodes_timeout_bytes_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="echo test",
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute(
        {"command": "echo test"},
        ToolContext(workspace=tmp_path, shell_timeout_seconds=1),
    )

    assert result.success is False
    assert result.payload["timed_out"] is True
    assert result.payload["stdout"] == "partial stdout"
    assert result.payload["stderr"] == "partial stderr"


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


def test_shell_tool_returns_error_when_process_launch_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(*args, **kwargs):
        raise OSError("workspace unavailable")

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute(
        {"command": "echo test"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Command launch failed"
    assert result.summary == "Command launch failed: OSError"
    assert result.payload["command"] == "echo test"
    assert result.payload["working_directory"] == str(tmp_path)
    assert result.payload["exit_code"] is None
    assert result.payload["timed_out"] is False
    assert result.payload["stderr"] == "workspace unavailable"


def test_shell_tool_returns_error_when_launch_error_is_unstringable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class UnstringableOSError(OSError):
        def __str__(self) -> str:
            raise RuntimeError("message unavailable")

    def fake_run(*args, **kwargs):
        raise UnstringableOSError()

    monkeypatch.setattr("agentskeleton.tools.shell.subprocess.run", fake_run)

    result = ShellTool().execute(
        {"command": "echo test"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Command launch failed"
    assert result.summary == "Command launch failed: UnstringableOSError"
    assert result.payload["stderr"] == "UnstringableOSError"
