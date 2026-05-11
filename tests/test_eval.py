from pathlib import Path

from agentskeleton.config import RunConfig
from agentskeleton.eval import (
    load_scenario,
    load_scenario_suite,
    load_scenario_suite_config,
    run_scenario,
    run_scenario_suite,
)
from agentskeleton.tools.filesystem import ReadFileTool, WriteFileTool
from agentskeleton.tools.loading import load_tools_from_modules
from agentskeleton.tools.registry import ToolRegistry
from agentskeleton.tools.user import AskUserTool


def test_run_scenario_executes_scripted_actions_and_checks_expectations(
    tmp_path: Path,
) -> None:
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

    scenario = load_scenario(scenario_path)
    result = run_scenario(
        scenario,
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is True
    assert result.failures == []
    assert result.scenario == "read-note"
    assert result.status == "completed"
    assert result.answer == "read complete"
    assert result.observations == 1
    assert result.log_path.exists()


def test_run_scenario_includes_domain_metadata(tmp_path: Path) -> None:
    scenario_path = tmp_path / "finance.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: finance-read",
                "domain: finance",
                "goal: inspect an invoice",
                "actions:",
                "  - type: final",
                "    text: finance ok",
                "expect:",
                "  status: completed",
                "  answer: finance ok",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.domain == "finance"
    assert result.to_dict()["domain"] == "finance"


def test_run_scenario_applies_declared_workspace_files(tmp_path: Path) -> None:
    scenario_path = tmp_path / "fixture-read.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: fixture-read",
                "goal: read fixture",
                "files:",
                "  data/input.txt: fixture content",
                "actions:",
                "  - type: tool",
                "    tool: read_file",
                "    call_id: read-1",
                "    arguments:",
                "      path: data/input.txt",
                "  - type: final",
                "    text: fixture read ok",
                "expect:",
                "  status: completed",
                "  answer: fixture read ok",
                "  observations: 1",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is True
    assert result.workspace != tmp_path.resolve()
    assert (result.workspace / "data" / "input.txt").read_text(
        encoding="utf-8"
    ) == "fixture content"
    assert not (tmp_path / "data" / "input.txt").exists()


def test_run_scenario_preserves_lf_newlines_in_declared_files(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "fixture-newlines.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: fixture-newlines",
                "goal: read fixture newlines",
                "files:",
                "  data/input.txt: |",
                "    first",
                "    second",
                "actions:",
                "  - type: final",
                "    text: fixture newlines ok",
                "expect:",
                "  status: completed",
                "  answer: fixture newlines ok",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is True
    assert (result.workspace / "data" / "input.txt").read_bytes() == (
        b"first\nsecond\n"
    )


def test_run_scenario_reports_failed_expectations(tmp_path: Path) -> None:
    scenario_path = tmp_path / "mismatch.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "goal: finish",
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

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is False
    assert result.failures == ["answer expected 'expected' but got 'actual'"]


def test_run_scenario_checks_expected_file_outputs(tmp_path: Path) -> None:
    scenario_path = tmp_path / "write-report.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: write-report",
                "goal: write report",
                "files:",
                "  inputs/source.txt: raw data",
                "actions:",
                "  - type: tool",
                "    tool: write_file",
                "    call_id: write-1",
                "    arguments:",
                "      path: reports/summary.txt",
                "      content: summary from raw data",
                "      create_parent_dirs: true",
                "  - type: final",
                "    text: report written",
                "expect:",
                "  status: completed",
                "  answer: report written",
                "  observations: 1",
                "  files:",
                "    reports/summary.txt: summary from raw data",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(
            workspace=tmp_path,
            logs_dir=tmp_path / "runs",
            permission_profile="trusted",
        ),
        ToolRegistry([WriteFileTool()]),
    )

    assert result.passed is True
    assert result.failures == []


def test_run_scenario_executes_batch_tool_actions(tmp_path: Path) -> None:
    scenario_path = tmp_path / "batch-read.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: batch-read",
                "goal: read two fixture files",
                "files:",
                "  data/one.txt: one",
                "  data/two.txt: two",
                "actions:",
                "  - type: batch",
                "    calls:",
                "      - tool: read_file",
                "        call_id: read-one",
                "        arguments:",
                "          path: data/one.txt",
                "      - tool: read_file",
                "        call_id: read-two",
                "        arguments:",
                "          path: data/two.txt",
                "  - type: final",
                "    text: batch read ok",
                "expect:",
                "  status: completed",
                "  answer: batch read ok",
                "  observations: 2",
                "  observations_detail:",
                "    - tool: read_file",
                "      success: true",
                "      payload:",
                "        path: data/one.txt",
                "        content: one",
                "    - tool: read_file",
                "      success: true",
                "      payload:",
                "        path: data/two.txt",
                "        content: two",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is True
    assert [failure for failure in result.failures] == []


def test_run_scenario_executes_invalid_batch_member(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "invalid-batch-member.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: invalid-batch-member",
                "goal: handle an invalid batch member",
                "actions:",
                "  - type: invalid_batch_member",
                "expect:",
                "  status: invalid_action",
                "  reason: 'Model returned unsupported action: dict'",
                "  observations: 0",
                "  events:",
                "    - type: run_error",
                "      payload:",
                "        status: invalid_action",
                "        error_type: dict",
                "        error: 'Model returned unsupported action: dict'",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is True
    assert result.failures == []


def test_run_scenario_uses_scripted_user_answers(tmp_path: Path) -> None:
    scenario_path = tmp_path / "interactive.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: interactive",
                "goal: clarify the customer segment",
                "user_answers:",
                "  - enterprise",
                "actions:",
                "  - type: tool",
                "    tool: ask_user",
                "    call_id: ask-segment",
                "    arguments:",
                "      question: Which customer segment?",
                "  - type: final",
                "    text: interactive ok",
                "expect:",
                "  status: completed",
                "  answer: interactive ok",
                "  observations: 1",
                "  observations_detail:",
                "    - tool: ask_user",
                "      success: true",
                "      summary: User answered question",
                "      payload:",
                "        question: Which customer segment?",
                "        answer: enterprise",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([AskUserTool()]),
    )

    assert result.passed is True
    assert result.failures == []


def test_run_scenario_starts_with_declared_conversation(tmp_path: Path) -> None:
    scenario_path = tmp_path / "resume-context.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: resume-context",
                "goal: continue the prior task",
                "conversation:",
                "  - role: user",
                "    content: remember alpha",
                "  - role: assistant",
                "    content: alpha stored",
                "actions:",
                "  - type: final",
                "    text: resumed ok",
                "expect:",
                "  status: completed",
                "  answer: resumed ok",
                "  events:",
                "    - type: run_started",
                "      payload:",
                "        resumed: true",
                "        conversation_turns: 2",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is True
    assert result.failures == []


def test_run_scenario_executes_scripted_model_error_and_checks_reason(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "model-error.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: model-error",
                "goal: handle a model outage",
                "actions:",
                "  - type: error",
                "    message: model unavailable",
                "expect:",
                "  status: model_error",
                "  reason: 'Model call failed: RuntimeError'",
                "  observations: 0",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is True
    assert result.reason == "Model call failed: RuntimeError"
    assert result.to_dict()["reason"] == "Model call failed: RuntimeError"
    assert result.failures == []


def test_run_scenario_applies_config_overrides(tmp_path: Path) -> None:
    scenario_path = tmp_path / "trusted-write.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: trusted-write",
                "goal: write trusted report",
                "config:",
                "  permission_profile: trusted",
                "files:",
                "  inputs/source.txt: raw data",
                "actions:",
                "  - type: tool",
                "    tool: write_file",
                "    call_id: write-1",
                "    arguments:",
                "      path: reports/summary.txt",
                "      content: summary from raw data",
                "      create_parent_dirs: true",
                "  - type: final",
                "    text: report written",
                "expect:",
                "  status: completed",
                "  answer: report written",
                "  observations: 1",
                "  files:",
                "    reports/summary.txt: summary from raw data",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([WriteFileTool()]),
    )

    assert result.passed is True
    assert result.status == "completed"
    assert result.failures == []


def test_run_scenario_uses_registry_factory_after_config_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "custom_tools.py").write_text(
        "\n".join(
            [
                "from agentskeleton.tools.base import Tool, ToolResult",
                "",
                "class ClassifyTool(Tool):",
                "    name = 'classify_domain'",
                "    description = 'Classify a domain.'",
                "    risk = 'read'",
                "    args_schema = {",
                "        'type': 'object',",
                "        'properties': {'text': {'type': 'string'}},",
                "        'required': ['text'],",
                "        'additionalProperties': False,",
                "    }",
                "    def execute(self, args, context):",
                "        return ToolResult(",
                "            success=True,",
                "            payload={'domain': 'finance'},",
                "            summary='classified finance',",
                "        )",
                "",
                "TOOLS = [ClassifyTool()]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
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
                "    arguments:",
                "      text: reconcile Q4 invoice anomalies",
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

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        registry_factory=lambda config: ToolRegistry(
            load_tools_from_modules(config.tool_modules)
        ),
    )

    assert result.passed is True
    assert result.status == "completed"


def test_run_scenario_checks_observation_details(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "custom_tools.py").write_text(
        "\n".join(
            [
                "from agentskeleton.tools.base import Tool, ToolResult",
                "",
                "class ClassifyTool(Tool):",
                "    name = 'classify_domain'",
                "    description = 'Classify a domain.'",
                "    risk = 'read'",
                "    args_schema = {",
                "        'type': 'object',",
                "        'properties': {'text': {'type': 'string'}},",
                "        'required': ['text'],",
                "        'additionalProperties': False,",
                "    }",
                "    def execute(self, args, context):",
                "        return ToolResult(",
                "            success=True,",
                "            payload={'domain': 'finance', 'confidence': 'high'},",
                "            summary='classified finance',",
                "        )",
                "",
                "TOOLS = [ClassifyTool()]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
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
                "    arguments:",
                "      text: reconcile Q4 invoice anomalies",
                "  - type: final",
                "    text: classified",
                "expect:",
                "  status: completed",
                "  answer: classified",
                "  observations: 1",
                "  observations_detail:",
                "    - tool: classify_domain",
                "      success: true",
                "      summary: classified finance",
                "      payload:",
                "        domain: finance",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        registry_factory=lambda config: ToolRegistry(
            load_tools_from_modules(config.tool_modules)
        ),
    )

    assert result.passed is True
    assert result.failures == []


def test_run_scenario_reports_observation_detail_mismatch(tmp_path: Path) -> None:
    scenario_path = tmp_path / "invalid-read.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: invalid-read",
                "goal: call read_file incorrectly",
                "actions:",
                "  - type: tool",
                "    tool: read_file",
                "    call_id: read-1",
                "    arguments: {}",
                "  - type: final",
                "    text: recovered",
                "expect:",
                "  status: completed",
                "  observations: 1",
                "  observations_detail:",
                "    - tool: read_file",
                "      success: true",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is False
    assert result.failures == ["observation 1 success expected True but got False"]


def test_run_scenario_checks_expected_log_events(tmp_path: Path) -> None:
    scenario_path = tmp_path / "events.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: events",
                "goal: finish with logged events",
                "actions:",
                "  - type: final",
                "    text: done",
                "expect:",
                "  status: completed",
                "  answer: done",
                "  events:",
                "    - type: run_started",
                "      payload:",
                "        goal: finish with logged events",
                "    - type: run_finished",
                "      payload:",
                "        status: completed",
                "        answer: done",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is True
    assert result.failures == []


def test_run_scenario_reports_log_event_mismatch(tmp_path: Path) -> None:
    scenario_path = tmp_path / "event-mismatch.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: event-mismatch",
                "goal: finish",
                "actions:",
                "  - type: final",
                "    text: done",
                "expect:",
                "  status: completed",
                "  events:",
                "    - type: run_finished",
                "      payload:",
                "        status: failed",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is False
    assert result.failures == [
        "event 1 expected {'type': 'run_finished', 'payload': {'status': 'failed'}} "
        "but was not found"
    ]


def test_run_scenario_reports_expected_file_mismatch(tmp_path: Path) -> None:
    scenario_path = tmp_path / "missing-report.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: missing-report",
                "goal: skip report",
                "files:",
                "  inputs/source.txt: raw data",
                "actions:",
                "  - type: final",
                "    text: skipped",
                "expect:",
                "  status: completed",
                "  files:",
                "    reports/summary.txt: summary from raw data",
            ]
        ),
        encoding="utf-8",
    )

    result = run_scenario(
        load_scenario(scenario_path),
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([WriteFileTool()]),
    )

    assert result.passed is False
    assert result.failures == ["file reports/summary.txt expected but was missing"]


def test_load_scenario_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    try:
        load_scenario(missing_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file not found: {missing_path}"
    else:
        raise AssertionError("Expected missing scenario file to fail")


def test_load_scenario_requires_expectations(tmp_path: Path) -> None:
    scenario_path = tmp_path / "unchecked.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "goal: finish",
                "actions:",
                "  - type: final",
                "    text: done",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == "Scenario must define expectations"
    else:
        raise AssertionError("Expected unchecked scenario to fail")


def test_load_scenario_suite_discovers_yaml_files_in_order(tmp_path: Path) -> None:
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    (suite_dir / "b.yaml").write_text(
        "\n".join(
            [
                "goal: b",
                "actions:",
                "  - type: final",
                "    text: b",
                "expect:",
                "  status: completed",
            ]
        ),
        encoding="utf-8",
    )
    (suite_dir / "a.yml").write_text(
        "\n".join(
            [
                "goal: a",
                "actions:",
                "  - type: final",
                "    text: a",
                "expect:",
                "  status: completed",
            ]
        ),
        encoding="utf-8",
    )
    (suite_dir / "ignored.txt").write_text("ignore", encoding="utf-8")

    scenarios = load_scenario_suite(suite_dir)

    assert [scenario.name for scenario in scenarios] == ["a", "b"]


def test_load_scenario_suite_config_reads_manifest_required_domains(
    tmp_path: Path,
) -> None:
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

    config = load_scenario_suite_config(suite_dir)
    scenarios = load_scenario_suite(suite_dir)

    assert config.required_domains == ["finance", "writing"]
    assert [scenario.name for scenario in scenarios] == ["alpha"]


def test_run_scenario_suite_summarizes_passes_and_failures(tmp_path: Path) -> None:
    passing = load_scenario(
        _write_final_scenario(
            tmp_path,
            "passing",
            "ok",
            "ok",
            domain="finance",
        )
    )
    failing = load_scenario(
        _write_final_scenario(
            tmp_path,
            "failing",
            "actual",
            "expected",
            domain="writing",
        )
    )

    suite = run_scenario_suite(
        [passing, failing],
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert suite.passed is False
    assert suite.total == 2
    assert suite.passed_count == 1
    assert suite.failed_count == 1
    assert suite.domains == {
        "finance": {"passed": 1, "failed": 0, "total": 1},
        "writing": {"passed": 0, "failed": 1, "total": 1},
    }
    assert suite.to_dict()["domains"] == suite.domains
    assert [result.scenario for result in suite.results] == ["passing", "failing"]
    assert suite.results[1].failures == [
        "answer expected 'expected' but got 'actual'"
    ]


def test_run_scenario_suite_fails_when_required_domain_is_missing(
    tmp_path: Path,
) -> None:
    scenario = load_scenario(
        _write_final_scenario(
            tmp_path,
            "passing",
            "ok",
            "ok",
            domain="finance",
        )
    )

    suite = run_scenario_suite(
        [scenario],
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
        required_domains=["finance", "writing"],
    )

    assert suite.passed is False
    assert suite.coverage_failures == ["required domain writing has no scenarios"]
    assert suite.to_dict()["coverage_failures"] == suite.coverage_failures


def test_run_scenario_suite_fails_when_required_domain_has_failures(
    tmp_path: Path,
) -> None:
    scenario = load_scenario(
        _write_final_scenario(
            tmp_path,
            "failing",
            "actual",
            "expected",
            domain="finance",
        )
    )

    suite = run_scenario_suite(
        [scenario],
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
        required_domains=["finance"],
    )

    assert suite.passed is False
    assert suite.coverage_failures == ["required domain finance has failing scenarios"]


def _write_final_scenario(
    tmp_path: Path,
    name: str,
    final_text: str,
    expected_answer: str,
    domain: str | None = None,
) -> Path:
    domain_lines = [f"domain: {domain}"] if domain else []
    scenario_path = tmp_path / f"{name}.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                f"name: {name}",
                *domain_lines,
                f"goal: {name}",
                "actions:",
                "  - type: final",
                f"    text: {final_text}",
                "expect:",
                "  status: completed",
                f"  answer: {expected_answer}",
            ]
        ),
        encoding="utf-8",
    )
    return scenario_path
