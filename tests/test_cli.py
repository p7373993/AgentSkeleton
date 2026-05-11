import json

from typer.testing import CliRunner

from agentskeleton.cli import app, build_default_registry, configure_streams_for_unicode
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


def test_default_registry_can_filter_enabled_tools() -> None:
    registry = build_default_registry(["read_file", "ask_user"])

    assert [tool.name for tool in registry.all()] == ["read_file", "ask_user"]


def test_default_registry_rejects_unknown_enabled_tool() -> None:
    try:
        build_default_registry(["read_file", "missing"])
    except ValueError as exc:
        assert str(exc) == "Unknown enabled tool: missing"
    else:
        raise AssertionError("Expected unknown enabled tool to fail")


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


def test_run_output_includes_run_id_and_final_reason(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    monkeypatch.setattr("agentskeleton.cli.uuid4", lambda: "run-fixed")

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            pass

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
            return type(
                "State",
                (),
                {
                    "final_status": "model_error",
                    "final_answer": None,
                    "final_reason": "Model call failed: RuntimeError",
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
    assert "Run id: run-fixed" in result.stdout
    assert "Status: model_error" in result.stdout
    assert "Reason: Model call failed: RuntimeError" in result.stdout
    assert "Run log:" in result.stdout


def test_run_uses_enabled_tools_from_config(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "\n".join(
            [
                "enabled_tools:",
                "  - read_file",
                "  - ask_user",
            ]
        ),
        encoding="utf-8",
    )
    seen_tools: list[list[str]] = []

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            seen_tools.append([tool.name for tool in kwargs["registry"].all()])

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
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

    result = runner.invoke(app, ["run", "continue", "--config", str(config_path)])

    assert result.exit_code == 0
    assert seen_tools == [["read_file", "ask_user"]]


def test_run_tool_option_overrides_enabled_tools_config(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "\n".join(
            [
                "enabled_tools:",
                "  - shell",
            ]
        ),
        encoding="utf-8",
    )
    seen_tools: list[list[str]] = []

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            seen_tools.append([tool.name for tool in kwargs["registry"].all()])

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
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

    result = runner.invoke(
        app,
        [
            "run",
            "continue",
            "--config",
            str(config_path),
            "--tool",
            "read_file",
            "--tool",
            "ask_user",
        ],
    )

    assert result.exit_code == 0
    assert seen_tools == [["read_file", "ask_user"]]


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


def test_resume_reuses_named_session_transcript(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    store = SessionStore(tmp_path / "runs")
    store.append_transcript("work", "user", "remember beta")
    store.append_transcript("work", "assistant", "beta stored")
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
            assert trace_context == {"session": "work"}
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

    result = runner.invoke(app, ["resume", "work"], input="next\n/exit\n")

    assert result.exit_code == 0
    assert seen_conversation == [["remember beta", "beta stored"]]
    session = SessionStore(tmp_path / "runs").load("work")
    assert [turn.content for turn in session.transcript] == [
        "remember beta",
        "beta stored",
        "next",
        "answer: next",
    ]
    assert "assistant> answer: next" in result.stdout


def test_show_run_prints_summary_from_jsonl_log(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
        {
            "type": "run_started",
            "run_id": "run-1",
            "step": 0,
            "payload": {"goal": "finish"},
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 3,
            "payload": {
                "status": "completed",
                "answer": "done",
                "reason": "finished cleanly",
            },
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1"])

    assert result.exit_code == 0
    assert "Run id: run-1" in result.stdout
    assert "Status: completed" in result.stdout
    assert "Reason: finished cleanly" in result.stdout
    assert "Steps: 3" in result.stdout
    assert f"Log: {log_path}" in result.stdout


def test_show_run_reports_missing_run(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "missing"])

    assert result.exit_code == 1
    assert "Run log not found: missing" in result.stdout
