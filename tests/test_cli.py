import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import agentskeleton.cli as cli_module
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


def test_eval_command_bounds_large_reason_and_failure_output(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    scenario_path = tmp_path / "large-output.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: large-output",
                "goal: large output",
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
    large_reason = "r" * 20_000
    large_failure = "f" * 20_000

    class FakeResult:
        passed = False

        def to_dict(self) -> dict[str, object]:
            return {
                "scenario": "large-output",
                "passed": False,
                "status": "model_error",
                "reason": large_reason,
                "failures": [large_failure],
                "log": str(tmp_path / "large-output.jsonl"),
            }

    monkeypatch.setattr(
        "agentskeleton.cli.run_scenario",
        lambda *args, **kwargs: FakeResult(),
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eval", str(scenario_path)])

    assert result.exit_code == 1
    assert result.stdout.count("[truncated") >= 2
    assert large_reason not in result.stdout
    assert large_failure not in result.stdout


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


def test_eval_suite_command_bounds_large_reason_and_failure_output(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    (suite_dir / "large-output.yaml").write_text(
        "\n".join(
            [
                "name: large-output",
                "goal: large output",
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
    large_reason = "r" * 20_000
    large_failure = "f" * 20_000
    large_coverage_failure = "c" * 20_000

    class FakeResult:
        passed = False

        def to_dict(self) -> dict[str, object]:
            return {
                "passed": False,
                "passed_count": 0,
                "total": 1,
                "results": [
                    {
                        "scenario": "large-output",
                        "passed": False,
                        "status": "model_error",
                        "reason": large_reason,
                        "failures": [large_failure],
                    }
                ],
                "coverage_failures": [large_coverage_failure],
            }

    monkeypatch.setattr(
        "agentskeleton.cli.run_scenario_suite",
        lambda *args, **kwargs: FakeResult(),
    )
    runner = CliRunner()

    result = runner.invoke(app, ["eval-suite", str(suite_dir)])

    assert result.exit_code == 1
    assert result.stdout.count("[truncated") >= 3
    assert large_reason not in result.stdout
    assert large_failure not in result.stdout
    assert large_coverage_failure not in result.stdout


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


def test_merge_required_domains_accepts_option_domains_without_length() -> None:
    class ExplodingDomainList(list):
        def __len__(self) -> int:
            raise RuntimeError("domain count unavailable")

    assert cli_module._merge_required_domains(  # noqa: SLF001
        ["finance"],
        ExplodingDomainList(["finance", "writing"]),
    ) == ["finance", "writing"]


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


def test_cli_reports_workspace_file_config_error(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_file = tmp_path / "workspace"
    workspace_file.write_text("not a directory", encoding="utf-8")
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("workspace: workspace\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Configuration error:" in result.stdout
    assert "workspace must be a directory path" in result.stdout


def test_cli_reports_blank_model_config_error(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text('model: "   "\n', encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Configuration error:" in result.stdout
    assert "model cannot be blank" in result.stdout


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


def test_default_registry_rejects_unstringable_unknown_enabled_tool() -> None:
    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("tool name unavailable")

    try:
        build_default_registry([UnstringableString("missing")])
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


def test_run_rejects_blank_goal_before_api_key_check(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(app, ["run", "   "])

    assert result.exit_code == 1
    assert "Goal error: Goal cannot be blank" in result.stdout
    assert "OPENAI_API_KEY" not in result.stdout


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


def test_run_bounds_large_final_answer_output(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    large_answer = "a" * 20_000

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
                    "final_status": "completed",
                    "final_answer": large_answer,
                    "final_reason": None,
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["run", "finish"])

    assert result.exit_code == 0
    assert "a" * 40 in result.stdout
    assert "[truncated" in result.stdout
    assert large_answer not in result.stdout


def test_run_output_handles_uninspectable_final_answer(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")

    class UninspectableValue:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

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
                    "final_status": "completed",
                    "final_answer": UninspectableValue(),
                    "final_reason": None,
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["run", "finish"])

    assert result.exit_code == 0
    assert "<uninspectable>" in result.stdout


def test_run_output_handles_lengthless_final_answer(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")

    class ExplodingText(str):
        def __len__(self) -> int:
            raise RuntimeError("answer length unavailable")

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
                    "final_status": "completed",
                    "final_answer": ExplodingText("done"),
                    "final_reason": None,
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["run", "finish"])

    assert result.exit_code == 0
    assert "done" in result.stdout


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


def test_run_bounds_large_final_reason_output(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    large_reason = "r" * 20_000

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
                    "final_reason": large_reason,
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
    assert "r" * 40 in result.stdout
    assert "[truncated" in result.stdout
    assert large_reason not in result.stdout


def test_assistant_transcript_content_bounds_large_text() -> None:
    large_answer = "a" * 20_000
    answer_state = type(
        "State",
        (),
        {
            "final_status": "completed",
            "final_answer": large_answer,
            "final_reason": None,
        },
    )()

    answer_content = cli_module._assistant_transcript_content(answer_state)

    assert answer_content is not None
    assert len(answer_content) < 5_000
    assert answer_content.startswith("aaaaaaaaaaaaaaaa")
    assert "[truncated" in answer_content
    assert large_answer not in answer_content

    large_reason = "r" * 20_000
    reason_state = type(
        "State",
        (),
        {
            "final_status": "model_error",
            "final_answer": None,
            "final_reason": large_reason,
        },
    )()

    reason_content = cli_module._assistant_transcript_content(reason_state)

    assert reason_content is not None
    assert len(reason_content) < 5_000
    assert reason_content.startswith("Run stopped with status model_error.")
    assert "[truncated" in reason_content
    assert large_reason not in reason_content


def test_assistant_transcript_content_handles_uninspectable_values() -> None:
    class UninspectableValue:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    value = UninspectableValue()
    answer_state = type(
        "State",
        (),
        {
            "final_status": "completed",
            "final_answer": value,
            "final_reason": None,
        },
    )()

    answer_content = cli_module._assistant_transcript_content(answer_state)

    assert answer_content == "<uninspectable>"

    reason_state = type(
        "State",
        (),
        {
            "final_status": "model_error",
            "final_answer": None,
            "final_reason": value,
        },
    )()

    reason_content = cli_module._assistant_transcript_content(reason_state)

    assert reason_content == (
        "Run stopped with status model_error. Reason: <uninspectable>"
    )


def test_assistant_transcript_content_handles_lengthless_text_values() -> None:
    class ExplodingText(str):
        def __len__(self) -> int:
            raise RuntimeError("text length unavailable")

    answer_state = type(
        "State",
        (),
        {
            "final_status": "completed",
            "final_answer": ExplodingText("done"),
            "final_reason": None,
        },
    )()

    answer_content = cli_module._assistant_transcript_content(answer_state)

    assert answer_content == "done"

    reason_state = type(
        "State",
        (),
        {
            "final_status": ExplodingText("model_error"),
            "final_answer": None,
            "final_reason": ExplodingText("Model call failed"),
        },
    )()

    reason_content = cli_module._assistant_transcript_content(reason_state)

    assert reason_content == (
        "Run stopped with status model_error. Reason: Model call failed"
    )


def test_assistant_transcript_metadata_bounds_large_reason() -> None:
    large_reason = "r" * 20_000
    state = type(
        "State",
        (),
        {
            "final_status": "model_error",
            "final_reason": large_reason,
        },
    )()

    metadata = cli_module._assistant_transcript_metadata("run-fixed", state)

    assert metadata["run_id"] == "run-fixed"
    assert metadata["status"] == "model_error"
    assert isinstance(metadata["reason"], str)
    assert len(metadata["reason"]) < 5_000
    assert str(metadata["reason"]).startswith("rrrrrrrrrrrrrrrr")
    assert "[truncated" in str(metadata["reason"])
    assert large_reason not in str(metadata["reason"])


def test_assistant_transcript_metadata_handles_uninspectable_reason() -> None:
    class UninspectableValue:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    state = type(
        "State",
        (),
        {
            "final_status": "model_error",
            "final_reason": UninspectableValue(),
        },
    )()

    metadata = cli_module._assistant_transcript_metadata("run-fixed", state)

    assert metadata == {
        "run_id": "run-fixed",
        "status": "model_error",
        "reason": "<uninspectable>",
    }


def test_assistant_transcript_metadata_handles_lengthless_reason() -> None:
    class ExplodingText(str):
        def __len__(self) -> int:
            raise RuntimeError("reason length unavailable")

    state = type(
        "State",
        (),
        {
            "final_status": "model_error",
            "final_reason": ExplodingText("Model call failed"),
        },
    )()

    metadata = cli_module._assistant_transcript_metadata("run-fixed", state)

    assert metadata == {
        "run_id": "run-fixed",
        "status": "model_error",
        "reason": "Model call failed",
    }


def test_run_bounds_large_confirmation_prompt_output(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    large_reason = "r" * 20_000
    large_argument = "a" * 20_000

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            self.confirmer = kwargs["confirmer"]

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
            decision = type("Decision", (), {"reason": large_reason})()
            action = type(
                "Action",
                (),
                {
                    "tool_name": "write_file",
                    "arguments": {"payload": large_argument},
                },
            )()
            assert self.confirmer(decision, action) is False
            return type(
                "State",
                (),
                {
                    "final_status": "denied",
                    "final_answer": None,
                    "final_reason": "user denied",
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["run", "confirm"], input="n\n")

    assert result.exit_code == 0
    assert result.stdout.count("[truncated") >= 2
    assert large_reason not in result.stdout
    assert large_argument not in result.stdout


def test_run_confirmation_prompt_handles_uninspectable_values(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")

    class UninspectableValue:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

        def __repr__(self) -> str:
            raise RuntimeError("cannot repr")

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            self.confirmer = kwargs["confirmer"]

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
            value = UninspectableValue()
            decision = type("Decision", (), {"reason": value})()
            action = type(
                "Action",
                (),
                {
                    "tool_name": "write_file",
                    "arguments": {"payload": value},
                },
            )()
            assert self.confirmer(decision, action) is False
            return type(
                "State",
                (),
                {
                    "final_status": "denied",
                    "final_answer": None,
                    "final_reason": "user denied",
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["run", "confirm"], input="n\n")

    assert result.exit_code == 0
    assert "Reason: <uninspectable>" in result.stdout
    assert "Arguments: <uninspectable>" in result.stdout


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


def test_chat_bounds_large_final_answer_output(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    large_answer = "b" * 20_000

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
                    "final_status": "completed",
                    "final_answer": large_answer,
                    "final_reason": None,
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["chat"], input="finish\n/exit\n")

    assert result.exit_code == 0
    assert "b" * 40 in result.stdout
    assert "[truncated" in result.stdout
    assert large_answer not in result.stdout


def test_chat_output_handles_lengthless_final_answer(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")

    class ExplodingText(str):
        def __len__(self) -> int:
            raise RuntimeError("answer length unavailable")

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
                    "final_status": "completed",
                    "final_answer": ExplodingText("done"),
                    "final_reason": None,
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["chat"], input="finish\n/exit\n")

    assert result.exit_code == 0
    assert "assistant> done" in result.stdout


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


def test_chat_bounds_large_final_reason_output(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    large_reason = "r" * 20_000

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
                    "final_reason": large_reason,
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
    assert "r" * 40 in result.stdout
    assert "[truncated" in result.stdout
    assert large_reason not in result.stdout


def test_chat_output_handles_uninspectable_final_reason(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")

    class UninspectableValue:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

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
                    "final_reason": UninspectableValue(),
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
    assert "Reason: <uninspectable>" in result.stdout


def test_chat_bounds_large_confirmation_prompt_output(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    large_reason = "r" * 20_000
    large_argument = "a" * 20_000

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            self.confirmer = kwargs["confirmer"]

        def run(
            self,
            goal: str,
            conversation=None,
            trace_context: dict[str, object] | None = None,
        ):
            decision = type("Decision", (), {"reason": large_reason})()
            action = type(
                "Action",
                (),
                {
                    "tool_name": "write_file",
                    "arguments": {"payload": large_argument},
                },
            )()
            assert self.confirmer(decision, action) is False
            return type(
                "State",
                (),
                {
                    "final_status": "denied",
                    "final_answer": None,
                    "final_reason": "user denied",
                },
            )()

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: object(),
    )
    monkeypatch.setattr("agentskeleton.cli.AgentLoop", FakeLoop)
    runner = CliRunner()

    result = runner.invoke(app, ["chat"], input="first\nn\n/exit\n")

    assert result.exit_code == 0
    assert result.stdout.count("[truncated") >= 2
    assert large_reason not in result.stdout
    assert large_argument not in result.stdout


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


def test_sessions_command_reports_session_read_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    transcript_path = tmp_path / "runs" / "sessions" / "default" / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(
        json.dumps({"role": "user", "content": "hello"}),
        encoding="utf-8",
    )
    original_read_bytes = Path.read_bytes
    expected_path = transcript_path.resolve()

    def fail_read_bytes(path: Path) -> bytes:
        if path.resolve() == expected_path:
            raise OSError("permission denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    runner = CliRunner()

    result = runner.invoke(app, ["sessions"])

    assert result.exit_code == 1
    assert "Session error: Session transcript could not be read: default" in (
        result.stdout
    )


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


def test_show_run_text_preserves_rich_markup_literals(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
        {
            "type": "run_started",
            "run_id": "run-1",
            "step": 0,
            "payload": {"goal": "[bold]literal goal[/bold]"},
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {"status": "completed", "answer": "done"},
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1"])

    assert result.exit_code == 0
    assert "Goal: [bold]literal goal[/bold]" in result.stdout


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


def test_show_run_json_preserves_rich_markup_literals(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {
                "status": "completed",
                "answer": "[bold]literal answer[/bold]",
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
    assert payload["answer"] == "[bold]literal answer[/bold]"


def test_show_run_json_bounds_large_raw_log_text_values(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    large_answer = "a" * 20_000
    events = [
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {
                "status": "completed",
                "answer": large_answer,
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
    assert len(payload["answer"]) < 5_000
    assert payload["answer"].startswith("aaaaaaaaaaaaaaaa")
    assert "[truncated" in payload["answer"]
    assert large_answer not in result.stdout


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


def test_show_run_summarizes_bounded_tool_error_text_as_json(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    events = [
        {
            "type": "tool_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {
                "tool_name": "custom",
                "success": False,
                "summary": {
                    "truncated": True,
                    "bytes": 5_000,
                    "preview": "large summary...",
                },
                "error": {
                    "truncated": True,
                    "bytes": 6_000,
                    "preview": "large error...",
                },
            },
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {"status": "tool_error"},
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
    assert payload["last_tool_error"] == {
        "tool_name": "custom",
        "summary": "large summary... (truncated, 5000 bytes)",
        "error": "large error... (truncated, 6000 bytes)",
    }


def test_show_run_bounds_large_tool_error_name_as_json(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    large_tool_name = "t" * 20_000
    events = [
        {
            "type": "tool_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {
                "tool_name": large_tool_name,
                "success": False,
                "summary": "Command exited with 1",
                "error": "Command failed",
            },
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {"status": "tool_error"},
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
    tool_name = payload["last_tool_error"]["tool_name"]
    assert len(tool_name) < 5_000
    assert tool_name.startswith("tttttttttttttttt")
    assert "[truncated" in tool_name
    assert large_tool_name not in result.stdout


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


def test_show_run_events_bounds_large_raw_event_text(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    large_answer = "e" * 20_000
    events = [
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {
                "status": "completed",
                "answer": large_answer,
            },
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1", "--json", "--events"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    event_answer = payload["events"][0]["payload"]["answer"]
    assert len(event_answer) < 5_000
    assert event_answer.startswith("eeeeeeeeeeeeeeee")
    assert "[truncated" in event_answer
    assert large_answer not in result.stdout


def test_show_run_events_sanitize_unstringable_event_keys() -> None:
    class UnstringableKey:
        def __str__(self) -> str:
            raise RuntimeError("key unavailable")

    event = {UnstringableKey(): {"answer": "done"}}

    assert cli_module._bounded_run_log_event(event) == {
        "<uninspectable>": {"answer": "done"}
    }


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


def test_show_run_ignores_too_deep_jsonl_lines(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    too_deep = '{"child":' * 20_000 + "null" + "}" * 20_000
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
                too_deep,
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


def test_show_run_rejects_oversized_log_before_reading(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    log_path.write_bytes(b"x" * 2_097_153)
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1"])

    assert result.exit_code == 1
    assert "Run log error: Run log exceeds 2097152 bytes" in result.stdout


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


def test_list_runs_passes_limit_to_log_discovery(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    seen: dict[str, int | None] = {}
    log_path = tmp_path / "runs" / "20260511" / "run-1.jsonl"

    def fake_run_log_paths(logs_dir: Path, limit: int | None = None) -> list[Path]:
        seen["limit"] = limit
        return [log_path]

    def fake_summarize_run_log(log_path: Path) -> dict[str, object]:
        return {
            "run_id": "run-1",
            "goal": "finish",
            "workspace": None,
            "session": None,
            "resumed": None,
            "conversation_turns": None,
            "status": "completed",
            "reason": None,
            "answer": "done",
            "steps": 1,
            "tool_calls": 0,
            "tool_failures": 0,
            "last_tool_error": None,
            "log": str(log_path),
        }

    monkeypatch.setattr(cli_module, "_run_log_paths", fake_run_log_paths)
    monkeypatch.setattr(
        cli_module,
        "_summarize_run_log_or_exit",
        fake_summarize_run_log,
    )
    runner = CliRunner()

    result = runner.invoke(app, ["list-runs", "--limit", "1", "--json"])

    assert result.exit_code == 0
    assert seen["limit"] == 1


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


def test_list_runs_ignores_too_deep_jsonl_lines(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    too_deep = '{"child":' * 20_000 + "null" + "}" * 20_000
    (log_dir / "run-1.jsonl").write_text(
        "\n".join(
            [
                too_deep,
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


def test_show_run_rejects_log_that_grows_after_stat(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "runs" / "20260511"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "run-1.jsonl"
    log_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli_module, "MAX_RUN_LOG_BYTES", 10)
    original_read_bytes = Path.read_bytes
    expected_path = log_path.resolve()

    def grow_read_bytes(path: Path) -> bytes:
        if path.resolve() == expected_path:
            return b"x" * 11
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", grow_read_bytes)
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1"])

    assert result.exit_code == 1
    assert "Run log error: Run log exceeds 10 bytes:" in result.stdout


def test_show_run_reports_log_discovery_failures(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    original_glob = Path.glob

    def fail_run_log_glob(path: Path, pattern: str):
        if path == Path("runs") and pattern == "*/*.jsonl":
            raise OSError("permission denied")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", fail_run_log_glob)
    runner = CliRunner()

    result = runner.invoke(app, ["show-run", "run-1"])

    assert result.exit_code == 1
    assert "Run log error: Run log directory could not be read: runs" in result.stdout
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


def test_list_runs_reports_log_discovery_failures(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    original_glob = Path.glob

    def fail_run_log_glob(path: Path, pattern: str):
        if path == Path("runs") and pattern == "*/*.jsonl":
            raise OSError("permission denied")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", fail_run_log_glob)
    runner = CliRunner()

    result = runner.invoke(app, ["list-runs"])

    assert result.exit_code == 1
    assert "Run log error: Run log directory could not be read: runs" in result.stdout
    assert result.exception is None or not isinstance(result.exception, OSError)


def test_restore_run_reports_log_discovery_failures(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    original_glob = Path.glob

    def fail_run_log_glob(path: Path, pattern: str):
        if path == Path("runs") and pattern == "*/*.jsonl":
            raise OSError("permission denied")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", fail_run_log_glob)
    runner = CliRunner()

    result = runner.invoke(app, ["restore-run", "run-1"])

    assert result.exit_code == 1
    assert "Run log error: Run log directory could not be read: runs" in result.stdout
    assert result.exception is None or not isinstance(result.exception, OSError)


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


def test_restore_run_imports_bounded_log_text_previews(
    monkeypatch,
    tmp_path,
) -> None:
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
                "goal": {
                    "truncated": True,
                    "bytes": 5_000,
                    "preview": "summarize very long input...",
                }
            },
        },
        {
            "type": "run_finished",
            "run_id": "run-1",
            "step": 1,
            "payload": {
                "status": "completed",
                "answer": {
                    "truncated": True,
                    "bytes": 7_000,
                    "preview": "very long answer...",
                },
            },
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
        "summarize very long input... (truncated, 5000 bytes)",
        "very long answer... (truncated, 7000 bytes)",
    ]


def test_restore_run_reports_session_read_errors(monkeypatch, tmp_path) -> None:
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
    transcript_path = tmp_path / "runs" / "sessions" / "work" / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(
        json.dumps({"role": "user", "content": "existing"}),
        encoding="utf-8",
    )
    original_read_bytes = Path.read_bytes
    expected_path = transcript_path.resolve()

    def fail_read_bytes(path: Path) -> bytes:
        if path.resolve() == expected_path:
            raise OSError("permission denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    runner = CliRunner()

    result = runner.invoke(app, ["restore-run", "run-1", "--session", "work"])

    assert result.exit_code == 1
    assert "Session error: Session transcript could not be read: work" in (
        result.stdout
    )


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
