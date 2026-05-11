import json
from datetime import UTC, datetime

import pytest
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


def test_tools_command_uses_enabled_tools_from_config(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
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
    runner = CliRunner()

    result = runner.invoke(app, ["tools", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "read_file" in result.stdout
    assert "ask_user" in result.stdout
    assert "write_file" not in result.stdout
    assert "shell" not in result.stdout


def test_tools_command_tool_option_overrides_config(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
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
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "tools",
            "--config",
            str(config_path),
            "--tool",
            "read_file",
        ],
    )

    assert result.exit_code == 0
    assert "read_file" in result.stdout
    assert "shell" not in result.stdout


def test_tools_command_can_output_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
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
    runner = CliRunner()

    result = runner.invoke(app, ["tools", "--config", str(config_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [
        {"name": tool["name"], "risk": tool["risk"]} for tool in payload["tools"]
    ] == [
        {"name": "read_file", "risk": "read"},
        {"name": "ask_user", "risk": "interactive"},
    ]
    assert all(tool["description"] for tool in payload["tools"])


def test_doctor_can_validate_configured_tools_as_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "\n".join(
            [
                "logs_dir: logs",
                "enabled_tools:",
                "  - read_file",
                "  - ask_user",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--config", str(config_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "status": "ok",
        "workspace": str(tmp_path.resolve()),
        "logs_dir": "logs",
        "tool_count": 2,
        "tools": ["read_file", "ask_user"],
    }


def test_eval_command_runs_scenario_as_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    scenario_path = tmp_path / "read-note.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: read-note",
                "goal: read note.txt",
                "actions:",
                "  - type: tool",
                "    tool: read_file",
                "    call_id: read-1",
                "    arguments:",
                "      path: note.txt",
                "  - type: final",
                "    text: read complete",
                "expect:",
                "  status: completed",
                "  answer: read complete",
                "  observations: 1",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eval", str(scenario_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["scenario"] == "read-note"
    assert payload["passed"] is True
    assert payload["status"] == "completed"
    assert payload["answer"] == "read complete"
    assert payload["observations"] == 1
    assert payload["failures"] == []
    assert payload["log"].endswith(".jsonl")


def test_eval_command_prints_final_reason(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    scenario_path = tmp_path / "invalid-action.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: invalid-action",
                "goal: handle invalid action",
                "actions:",
                "  - type: invalid",
                "expect:",
                "  status: invalid_action",
                "  reason: 'Model returned unsupported action: dict'",
                "  observations: 0",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eval", str(scenario_path)])

    assert result.exit_code == 0
    assert "Scenario: invalid-action" in result.stdout
    assert "Status: invalid_action" in result.stdout
    assert "Reason: Model returned unsupported action: dict" in result.stdout


def test_eval_command_reports_directory_scenario_path(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    scenario_path = tmp_path / "directory.yaml"
    scenario_path.mkdir()
    runner = CliRunner()

    result = runner.invoke(app, ["eval", str(scenario_path)])

    assert result.exit_code == 1
    assert f"Scenario error: Scenario file must be a file: {scenario_path}" in (
        result.stdout
    )
    assert result.exception is None or not isinstance(result.exception, OSError)


def test_eval_suite_command_runs_directory_as_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    for name in ("alpha", "beta"):
        (suite_dir / f"{name}.yaml").write_text(
            "\n".join(
                [
                    f"name: {name}",
                    f"goal: {name}",
                    "actions:",
                    "  - type: final",
                    f"    text: {name} done",
                    "expect:",
                    "  status: completed",
                    f"  answer: {name} done",
                ]
            ),
            encoding="utf-8",
        )
    runner = CliRunner()

    result = runner.invoke(app, ["eval-suite", str(suite_dir), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["total"] == 2
    assert payload["passed_count"] == 2
    assert payload["failed_count"] == 0
    assert [item["scenario"] for item in payload["results"]] == ["alpha", "beta"]


def test_eval_suite_command_prints_status_and_reason(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    (suite_dir / "invalid-action.yaml").write_text(
        "\n".join(
            [
                "name: invalid-action",
                "goal: handle invalid action",
                "actions:",
                "  - type: invalid",
                "expect:",
                "  status: invalid_action",
                "  reason: 'Model returned unsupported action: dict'",
                "  observations: 0",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eval-suite", str(suite_dir)])

    assert result.exit_code == 0
    assert "PASS: invalid-action status=invalid_action" in result.stdout
    assert "Reason: Model returned unsupported action: dict" in result.stdout


def test_eval_suite_command_prints_scenario_failures(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    (suite_dir / "mismatch.yaml").write_text(
        "\n".join(
            [
                "name: mismatch",
                "goal: mismatch",
                "actions:",
                "  - type: final",
                "    text: actual",
                "expect:",
                "  status: completed",
                "  answer: expected",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eval-suite", str(suite_dir)])

    assert result.exit_code == 1
    assert "FAIL: mismatch status=completed" in result.stdout
    assert "Failure: answer expected 'expected' but got 'actual'" in result.stdout


def test_eval_suite_command_enforces_required_domains(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    (suite_dir / "alpha.yaml").write_text(
        "\n".join(
            [
                "name: alpha",
                "domain: finance",
                "goal: alpha",
                "actions:",
                "  - type: final",
                "    text: alpha done",
                "expect:",
                "  status: completed",
                "  answer: alpha done",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "eval-suite",
            str(suite_dir),
            "--require-domain",
            "finance",
            "--require-domain",
            "writing",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["coverage_failures"] == ["required domain writing has no scenarios"]


def test_eval_suite_command_prints_required_domain_failures(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    (suite_dir / "alpha.yaml").write_text(
        "\n".join(
            [
                "name: alpha",
                "domain: finance",
                "goal: alpha",
                "actions:",
                "  - type: final",
                "    text: alpha done",
                "expect:",
                "  status: completed",
                "  answer: alpha done",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "eval-suite",
            str(suite_dir),
            "--require-domain",
            "finance",
            "--require-domain",
            "writing",
        ],
    )

    assert result.exit_code == 1
    assert "Coverage failure: required domain writing has no scenarios" in result.stdout


def test_eval_suite_command_uses_manifest_required_domains(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    (suite_dir / "suite.yaml").write_text(
        "\n".join(
            [
                "required_domains:",
                "  - finance",
                "  - writing",
            ]
        ),
        encoding="utf-8",
    )
    (suite_dir / "alpha.yaml").write_text(
        "\n".join(
            [
                "name: alpha",
                "domain: finance",
                "goal: alpha",
                "actions:",
                "  - type: final",
                "    text: alpha done",
                "expect:",
                "  status: completed",
                "  answer: alpha done",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eval-suite", str(suite_dir), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["required_domains"] == ["finance", "writing"]
    assert payload["coverage_failures"] == ["required domain writing has no scenarios"]


def test_eval_command_uses_scenario_tool_modules(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "custom_tools.py").write_text(
        "\n".join(
            [
                "from agentskeleton.tools.base import Tool, ToolResult",
                "",
                "class ClassifyTool(Tool):",
                "    name = 'classify_domain'",
                "    description = 'Classify a domain.'",
                "    risk = 'read'",
                "    args_schema = {'type': 'object', 'properties': {}}",
                "    def execute(self, args, context):",
                "        return ToolResult(success=True, summary='classified')",
                "",
                "TOOLS = [ClassifyTool()]",
            ]
        ),
        encoding="utf-8",
    )
    scenario_path = tmp_path / "custom-domain.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: custom-domain",
                "goal: classify the request domain",
                "config:",
                "  tool_modules:",
                "    - custom_tools",
                "  enabled_tools:",
                "    - classify_domain",
                "actions:",
                "  - type: tool",
                "    tool: classify_domain",
                "    call_id: classify-1",
                "    arguments: {}",
                "  - type: final",
                "    text: classified",
                "expect:",
                "  status: completed",
                "  answer: classified",
                "  observations: 1",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eval", str(scenario_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["status"] == "completed"


def test_eval_command_reports_declared_file_parent_conflict(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    scenario_path = tmp_path / "fixture-conflict.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: fixture-conflict",
                "goal: prepare conflicting fixture files",
                "files:",
                "  reports: existing file",
                "  reports/summary.txt: child file",
                "actions:",
                "  - type: final",
                "    text: unreachable",
                "expect:",
                "  status: completed",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eval", str(scenario_path)])

    assert result.exit_code == 1
    assert (
        "Scenario error: Scenario file parent is not a directory: "
        "reports/summary.txt"
    ) in result.stdout
    assert result.exception is None or not isinstance(result.exception, OSError)


@pytest.mark.parametrize(
    ("command", "user_input"),
    [
        (["tools"], None),
        (["run", "finish"], None),
        (["chat"], "/exit\n"),
    ],
)
def test_cli_reports_registry_configuration_errors(
    command, user_input, monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "\n".join(
            [
                "enabled_tools:",
                "  - missing",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [*command, "--config", str(config_path)],
        input=user_input,
    )

    assert result.exit_code == 1
    assert "Configuration error: Unknown enabled tool: missing" in result.stdout
    assert result.exception is None or not isinstance(result.exception, ValueError)


def test_cli_reports_missing_config_file(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    missing_path = tmp_path / "missing.yaml"
    runner = CliRunner()

    result = runner.invoke(app, ["tools", "--config", str(missing_path)])

    assert result.exit_code == 1
    assert f"Configuration error: Config file not found: {missing_path}" in (
        result.stdout
    )


def test_cli_reports_config_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agent.yaml"
    config_path.mkdir()
    runner = CliRunner()

    result = runner.invoke(app, ["tools", "--config", str(config_path)])

    assert result.exit_code == 1
    assert f"Configuration error: Config file must be a file: {config_path}" in (
        result.stdout
    )
    assert result.exception is None or not isinstance(result.exception, OSError)


def test_cli_reports_logs_dir_file_config_error(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").write_text("not a directory", encoding="utf-8")
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("logs_dir: runs\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Configuration error:" in result.stdout
    assert "logs_dir must be a directory" in result.stdout


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


def test_default_registry_rejects_duplicate_tool_names_before_filtering(
    tmp_path,
    monkeypatch,
) -> None:
    module_path = tmp_path / "custom_tools.py"
    module_path.write_text(
        "\n".join(
            [
                "from agentskeleton.tools.base import Tool, ToolResult",
                "",
                "class DuplicateReadFileTool(Tool):",
                "    name = 'read_file'",
                "    description = 'Duplicate read file.'",
                "    risk = 'read'",
                "    args_schema = {'type': 'object', 'properties': {}}",
                "    def execute(self, args, context):",
                "        return ToolResult(success=True, summary='ok')",
                "",
                "TOOLS = [DuplicateReadFileTool()]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="Tool already registered: read_file"):
        build_default_registry(["read_file"], ["custom_tools"])


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


def test_run_reports_session_store_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / "runs" / "sessions" / "default"
    session_dir.parent.mkdir(parents=True)
    session_dir.write_text("not a directory", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["run", "finish"])

    assert result.exit_code == 1
    assert "Session error: Session path is not a directory: default" in result.stdout
    assert result.exception is None or not isinstance(result.exception, OSError)


def test_run_reports_log_directory_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    today = datetime.now(tz=UTC).strftime("%Y%m%d")
    log_dir = tmp_path / "runs"
    log_dir.mkdir()
    (log_dir / today).write_text("not a directory", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["run", "finish"])

    assert result.exit_code == 1
    assert "Run log error: Run log directory is not a directory:" in result.stdout
    assert result.exception is None or not isinstance(result.exception, OSError)


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


def test_run_refreshes_session_summary_for_long_transcript(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "session_context_turns: 2\nsession_summary_turns: 1\n",
        encoding="utf-8",
    )
    store = SessionStore(tmp_path / "runs")
    store.append_transcript("default", "user", "old user")
    store.append_transcript("default", "assistant", "old answer")
    store.append_transcript("default", "user", "recent user")
    store.append_transcript("default", "assistant", "recent answer")
    seen_conversation: list[list[tuple[str, str, dict[str, object]]]] = []

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            pass

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
            seen_conversation.append(
                [
                    (turn.role, turn.content, turn.metadata)
                    for turn in conversation or []
                ]
            )
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
    assert seen_conversation[0][0] == (
        "user",
        "Prior conversation summary:\n- assistant: old answer",
        {"source": "session_summary", "sticky_context": True},
    )
    assert [item[1] for item in seen_conversation[0][1:]] == [
        "old user",
        "old answer",
        "recent user",
        "recent answer",
    ]
    assert store.load("default").summary == "- assistant: old answer"


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


def test_run_persists_status_reason_when_no_final_answer(
    monkeypatch,
    tmp_path,
) -> None:
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
    session = SessionStore(tmp_path / "runs").load("default")
    assert [turn.role for turn in session.transcript] == ["user", "assistant"]
    assert [turn.content for turn in session.transcript] == [
        "continue",
        (
            "Run stopped with status model_error. "
            "Reason: Model call failed: RuntimeError"
        ),
    ]
    assert session.transcript[1].metadata == {
        "run_id": "run-fixed",
        "status": "model_error",
        "reason": "Model call failed: RuntimeError",
    }


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


def test_chat_prints_final_reason_when_no_answer(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")

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
                    "final_status": "invalid_action",
                    "final_answer": None,
                    "final_reason": "Model returned unsupported action: dict",
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["chat"], input="first\n/exit\n")

    assert result.exit_code == 0
    assert "Status: invalid_action" in result.stdout
    assert "Reason: Model returned unsupported action: dict" in result.stdout


def test_chat_persists_status_reason_for_next_turn(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
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
            if goal == "first":
                return type(
                    "State",
                    (),
                    {
                        "final_status": "invalid_action",
                        "final_answer": None,
                        "final_reason": "Model returned unsupported action: dict",
                    },
                )()
            return type(
                "State",
                (),
                {
                    "final_status": "completed",
                    "final_answer": "second answer",
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
        [],
        [
            "first",
            (
                "Run stopped with status invalid_action. "
                "Reason: Model returned unsupported action: dict"
            ),
        ],
    ]
    session = SessionStore(tmp_path / "runs").load("default")
    assert [turn.role for turn in session.transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert session.transcript[1].metadata["status"] == "invalid_action"
    assert (
        session.transcript[1].metadata["reason"]
        == "Model returned unsupported action: dict"
    )


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


def test_sessions_command_lists_sessions_as_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    store = SessionStore(tmp_path / "runs")
    store.append_transcript("default", "user", "hello")
    store.append_transcript("default", "assistant", "hi")
    store.append_transcript("work", "user", "old user")
    store.append_transcript("work", "assistant", "old answer")
    store.append_transcript("work", "user", "recent")
    store.refresh_summary("work", keep_turns=1)
    runner = CliRunner()

    result = runner.invoke(app, ["sessions", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "sessions": [
            {
                "name": "default",
                "transcript_turns": 2,
                "has_summary": False,
                "summary": None,
            },
            {
                "name": "work",
                "transcript_turns": 3,
                "has_summary": True,
                "summary": "- user: old user\n- assistant: old answer",
            },
        ]
    }


def test_sessions_command_prints_sessions(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    store = SessionStore(tmp_path / "runs")
    store.append_transcript("work", "user", "hello")
    runner = CliRunner()

    result = runner.invoke(app, ["sessions"])

    assert result.exit_code == 0
    assert "work" in result.stdout
    assert "1" in result.stdout


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
            "payload": {
                "goal": "finish",
                "workspace": str(tmp_path),
                "session": "work",
                "resumed": True,
                "conversation_turns": 2,
            },
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
    assert "Goal: finish" in result.stdout
    assert "Session: work" in result.stdout
    assert "Conversation turns: 2" in result.stdout
    assert "Steps: 3" in result.stdout
    assert f"Log: {log_path}" in result.stdout


def test_show_run_can_output_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
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

    result = runner.invoke(app, ["show-run", "run-1", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "run_id": "run-1",
        "goal": None,
        "workspace": None,
        "session": None,
        "resumed": None,
        "conversation_turns": None,
        "status": "completed",
        "reason": "finished cleanly",
        "answer": "done",
        "steps": 3,
        "tool_calls": 0,
        "tool_failures": 0,
        "last_tool_error": None,
        "log": str(log_path.resolve()),
    }


def test_show_run_summarizes_tool_results_as_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
        {
            "type": "run_started",
            "run_id": "run-1",
            "step": 0,
            "payload": {"goal": "debug tools"},
        },
        {
            "type": "tool_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {
                "tool_name": "read_file",
                "success": True,
                "summary": "Read 12 characters from README.md",
                "error": None,
            },
        },
        {
            "type": "tool_finished",
            "run_id": "run-1",
            "step": 2,
            "payload": {
                "tool_name": "shell",
                "success": False,
                "summary": "Command exited with 1",
                "error": "Command failed",
            },
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 2,
            "payload": {"status": "tool_error", "reason": "Command failed"},
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["tool_calls"] == 2
    assert payload["tool_failures"] == 1
    assert payload["last_tool_error"] == {
        "tool_name": "shell",
        "summary": "Command exited with 1",
        "error": "Command failed",
    }


def test_show_run_can_include_events_as_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
        {
            "type": "run_started",
            "run_id": "run-1",
            "step": 0,
            "payload": {"goal": "inspect timeline"},
        },
        {
            "type": "model_action",
            "run_id": "run-1",
            "step": 1,
            "payload": {"tool_name": "read_file", "arguments": {"path": "a.txt"}},
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {"status": "completed", "answer": "done"},
        },
    ]
    log_path.write_text(
        "\n".join(
            [
                json.dumps(events[0]),
                '{"type": "partial"',
                json.dumps(events[1]),
                json.dumps(events[2]),
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1", "--json", "--events"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["events"] == events


def test_show_run_ignores_malformed_jsonl_lines(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "run_started",
                        "run_id": "run-1",
                        "step": 0,
                        "payload": {"goal": "finish"},
                    }
                ),
                '{"type": "partial"',
                json.dumps(
                    {
                        "type": "run_finished",
                        "run_id": "run-1",
                        "step": 3,
                        "payload": {"status": "completed", "answer": "done"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["goal"] == "finish"
    assert payload["steps"] == 3


def test_show_run_ignores_non_object_jsonl_lines(monkeypatch, tmp_path) -> None:
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
            "payload": {"status": "completed", "answer": "done"},
        },
    ]
    log_path.write_text(
        "\n".join(
            [
                json.dumps(events[0]),
                json.dumps(["not", "an", "event"]),
                json.dumps("also not an event"),
                json.dumps(events[1]),
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1", "--json", "--events"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["goal"] == "finish"
    assert payload["events"] == events


def test_show_run_treats_non_object_event_payloads_as_empty(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "run_started",
                        "run_id": "run-1",
                        "step": 0,
                        "payload": ["not", "a", "mapping"],
                    }
                ),
                json.dumps(
                    {
                        "type": "run_finished",
                        "run_id": "run-1",
                        "step": 3,
                        "payload": "also not a mapping",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "run-1"
    assert payload["goal"] is None
    assert payload["status"] == "unknown"
    assert payload["steps"] == 3


def test_show_run_ignores_invalid_utf8_log_lines(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    log_path.write_bytes(
        json.dumps(
            {
                "type": "run_started",
                "run_id": "run-1",
                "step": 0,
                "payload": {"goal": "finish"},
            }
        ).encode("utf-8")
        + b"\n\xff\xfe\x00broken\n"
        + json.dumps(
            {
                "type": "run_finished",
                "run_id": "run-1",
                "step": 3,
                "payload": {"status": "completed", "answer": "done"},
            }
        ).encode("utf-8")
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["goal"] == "finish"
    assert payload["steps"] == 3


def test_list_runs_can_output_recent_runs_as_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    first_dir = tmp_path / "runs" / "20260510"
    second_dir = tmp_path / "runs" / "20260511"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    (first_dir / "run-old.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "run_started",
                        "run_id": "run-old",
                        "step": 0,
                        "payload": {"goal": "old goal"},
                    }
                ),
                json.dumps(
                    {
                        "type": "run_finished",
                        "run_id": "run-old",
                        "step": 2,
                        "payload": {"status": "completed", "answer": "old done"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (second_dir / "run-new.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "run_started",
                        "run_id": "run-new",
                        "step": 0,
                        "payload": {"goal": "new goal"},
                    }
                ),
                json.dumps(
                    {
                        "type": "run_finished",
                        "run_id": "run-new",
                        "step": 4,
                        "payload": {"status": "max_steps", "reason": "step limit"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["list-runs", "--limit", "1", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "runs": [
            {
                "run_id": "run-new",
                "goal": "new goal",
                "workspace": None,
                "session": None,
                "resumed": None,
                "conversation_turns": None,
                "status": "max_steps",
                "reason": "step limit",
                "answer": None,
                "steps": 4,
                "tool_calls": 0,
                "tool_failures": 0,
                "last_tool_error": None,
                "log": str((second_dir / "run-new.jsonl").resolve()),
            }
        ]
    }


def test_list_runs_ignores_malformed_jsonl_lines(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    (log_dir / "run-1.jsonl").write_text(
        "\n".join(
            [
                '{"type": "partial"',
                json.dumps(
                    {
                        "type": "run_finished",
                        "run_id": "run-1",
                        "step": 1,
                        "payload": {"status": "completed"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["list-runs", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["runs"][0]["run_id"] == "run-1"
    assert payload["runs"][0]["status"] == "completed"


def test_list_runs_prints_recent_runs(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    (log_dir / "run-1.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "run_started",
                        "run_id": "run-1",
                        "step": 0,
                        "payload": {"goal": "finish"},
                    }
                ),
                json.dumps(
                    {
                        "type": "run_finished",
                        "run_id": "run-1",
                        "step": 3,
                        "payload": {"status": "completed", "answer": "done"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["list-runs"])

    assert result.exit_code == 0
    assert "run-1" in result.stdout
    assert "completed" in result.stdout
    assert "finish" in result.stdout


def test_list_runs_prints_tool_failure_counts(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    (log_dir / "run-1.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "run_started",
                        "run_id": "run-1",
                        "step": 0,
                        "payload": {"goal": "debug failure"},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_finished",
                        "run_id": "run-1",
                        "step": 1,
                        "payload": {
                            "tool_name": "shell",
                            "success": False,
                            "summary": "Command exited with 1",
                            "error": "Command failed",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "run_finished",
                        "run_id": "run-1",
                        "step": 1,
                        "payload": {"status": "tool_error"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["list-runs"])

    assert result.exit_code == 0
    assert "Tool Failures" in result.stdout
    assert "1/1" in result.stdout


def test_show_run_reports_missing_run(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "missing"])

    assert result.exit_code == 1
    assert "Run log not found: missing" in result.stdout


def test_show_run_treats_run_id_as_literal_not_glob(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    (log_dir / "run-1.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "run_started",
                        "run_id": "run-1",
                        "step": 0,
                        "payload": {"goal": "summarize"},
                    }
                ),
                json.dumps(
                    {
                        "type": "run_finished",
                        "run_id": "run-1",
                        "step": 1,
                        "payload": {"status": "completed", "answer": "done"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "*"])

    assert result.exit_code == 1
    assert "Run log not found: *" in result.stdout


def test_show_run_reports_non_file_log_path(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    (log_dir / "run-1.jsonl").mkdir()
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1"])

    assert result.exit_code == 1
    assert "Run log error: Run log path is not a file:" in result.stdout
    assert result.exception is None or not isinstance(result.exception, OSError)


def test_list_runs_ignores_non_file_log_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    (log_dir / "run-1.jsonl").mkdir()
    runner = CliRunner()

    result = runner.invoke(app, ["list-runs"])

    assert result.exit_code == 0
    assert "No run logs found." in result.stdout


def test_restore_run_imports_run_log_into_session(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
        {
            "type": "run_started",
            "run_id": "run-1",
            "step": 0,
            "payload": {"goal": "finish the report"},
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 2,
            "payload": {
                "status": "max_steps",
                "reason": "Reached max_steps limit: 2",
                "answer": None,
            },
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["restore-run", "run-1", "--session", "work"],
    )

    assert result.exit_code == 0
    assert "Restored run run-1 into session work" in result.stdout
    session = SessionStore(tmp_path / "runs").load("work")
    assert [turn.role for turn in session.transcript] == ["user", "assistant"]
    assert [turn.content for turn in session.transcript] == [
        "finish the report",
        "Run stopped with status max_steps. Reason: Reached max_steps limit: 2",
    ]
    assert session.transcript[0].metadata == {
        "run_id": "run-1",
        "source": "run_log",
    }
    assert session.transcript[1].metadata == {
        "run_id": "run-1",
        "source": "run_log",
        "status": "max_steps",
        "reason": "Reached max_steps limit: 2",
    }


def test_restore_run_imports_final_answer(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
        {
            "type": "run_started",
            "run_id": "run-1",
            "step": 0,
            "payload": {"goal": "summarize"},
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {"status": "completed", "answer": "summary done"},
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["restore-run", "run-1"])

    assert result.exit_code == 0
    session = SessionStore(tmp_path / "runs").load("default")
    assert [turn.content for turn in session.transcript] == [
        "summarize",
        "summary done",
    ]
    assert session.transcript[1].metadata == {
        "run_id": "run-1",
        "source": "run_log",
        "status": "completed",
    }


def test_restore_run_does_not_duplicate_existing_import(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
        {
            "type": "run_started",
            "run_id": "run-1",
            "step": 0,
            "payload": {"goal": "summarize"},
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {"status": "completed", "answer": "summary done"},
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )
    runner = CliRunner()

    first = runner.invoke(app, ["restore-run", "run-1", "--session", "work"])
    second = runner.invoke(app, ["restore-run", "run-1", "--session", "work"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "already restored" in second.stdout
    session = SessionStore(tmp_path / "runs").load("work")
    assert [turn.content for turn in session.transcript] == [
        "summarize",
        "summary done",
    ]
