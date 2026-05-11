from typer.testing import CliRunner

from agentskeleton.cli import app, configure_streams_for_unicode
from agentskeleton.core.session import SessionStore


def test_tools_command_lists_default_tools() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["tools"])

    assert result.exit_code == 0
    assert "list_dir" in result.stdout
    assert "read_file" in result.stdout
    assert "write_file" in result.stdout
    assert "shell" in result.stdout
    assert "ask_user" in result.stdout


def test_configure_streams_for_unicode_uses_utf8_with_replacement() -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.calls = []

        def reconfigure(self, **kwargs) -> None:
            self.calls.append(kwargs)

    stdout = FakeStream()
    stderr = FakeStream()

    configure_streams_for_unicode(stdout, stderr)

    assert stdout.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_run_reports_missing_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(app, ["run", "finish"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required" in result.stdout


def test_run_reuses_default_session_transcript(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    store = SessionStore(tmp_path / "runs")
    store.append_transcript("default", "user", "remember alpha")
    store.append_transcript("default", "assistant", "alpha stored")
    seen_conversation: list[list[str]] = []

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            pass

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
            seen_conversation.append([turn.content for turn in conversation or []])
            return type(
                "State",
                (),
                {
                    "final_status": "completed",
                    "final_answer": "done",
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["run", "continue"])

    assert result.exit_code == 0
    assert seen_conversation == [["remember alpha", "alpha stored"]]
    session = SessionStore(tmp_path / "runs").load("default")
    assert [turn.role for turn in session.transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [turn.content for turn in session.transcript] == [
        "remember alpha",
        "alpha stored",
        "continue",
        "done",
    ]


def test_run_can_disable_session(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    store = SessionStore(tmp_path / "runs")
    store.append_transcript("default", "user", "remember alpha")
    seen_conversation: list[list[str]] = []

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            pass

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
            seen_conversation.append([turn.content for turn in conversation or []])
            return type(
                "State",
                (),
                {
                    "final_status": "completed",
                    "final_answer": "done",
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["run", "fresh", "--no-session"])

    assert result.exit_code == 0
    assert seen_conversation == [[]]
    session = SessionStore(tmp_path / "runs").load("default")
    assert [turn.content for turn in session.transcript] == ["remember alpha"]


def test_chat_reuses_session_transcript_between_inputs(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    store = SessionStore(tmp_path / "runs")
    store.append_transcript("default", "user", "remember alpha")
    store.append_transcript("default", "assistant", "alpha stored")
    seen_conversation: list[list[str]] = []

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            pass

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
            seen_conversation.append([turn.content for turn in conversation or []])
            return type(
                "State",
                (),
                {
                    "final_status": "completed",
                    "final_answer": f"answer: {goal}",
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["chat"], input="first\nsecond\n/exit\n")

    assert result.exit_code == 0
    assert seen_conversation == [
        ["remember alpha", "alpha stored"],
        ["remember alpha", "alpha stored", "first", "answer: first"],
    ]
    session = SessionStore(tmp_path / "runs").load("default")
    assert [turn.content for turn in session.transcript] == [
        "remember alpha",
        "alpha stored",
        "first",
        "answer: first",
        "second",
        "answer: second",
    ]
    assert "assistant> answer: first" in result.stdout
    assert "assistant> answer: second" in result.stdout


def test_chat_can_disable_session(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    store = SessionStore(tmp_path / "runs")
    store.append_transcript("default", "user", "remember alpha")
    seen_conversation: list[list[str]] = []

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            pass

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
            seen_conversation.append([turn.content for turn in conversation or []])
            return type(
                "State",
                (),
                {
                    "final_status": "completed",
                    "final_answer": "done",
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["chat", "--no-session"], input="first\n/exit\n")

    assert result.exit_code == 0
    assert seen_conversation == [[]]
    assert [turn.content for turn in store.load("default").transcript] == [
        "remember alpha"
    ]
