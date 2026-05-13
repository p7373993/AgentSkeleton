from pathlib import Path
from types import SimpleNamespace

import agentskeleton.eval as eval_module
from agentskeleton.config import RunConfig
from agentskeleton.eval import (
    _read_log_events,
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


class UnencodableString(str):
    def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("cannot encode")


class UninspectableString(str):
    def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("cannot strip")


class UnstringableValue:
    def __str__(self) -> str:
        raise RuntimeError("cannot stringify")


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


def test_run_scenario_expands_repeated_declared_workspace_files(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "fixture-repeat.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: fixture-repeat",
                "goal: prepare repeated fixture",
                "files:",
                "  data/repeated.txt:",
                "    repeat: ab",
                "    count: 3",
                "actions:",
                "  - type: final",
                "    text: fixture repeat ok",
                "expect:",
                "  status: completed",
                "  answer: fixture repeat ok",
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
    assert (result.workspace / "data" / "repeated.txt").read_text(
        encoding="utf-8"
    ) == "ababab"


def test_load_scenario_rejects_repeated_workspace_file_over_limit(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "fixture-repeat-large.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: fixture-repeat-large",
                "goal: prepare oversized repeated fixture",
                "files:",
                "  data/large.txt:",
                "    repeat: a",
                "    count: 2097153",
                "actions:",
                "  - type: final",
                "    text: unreachable",
                "expect:",
                "  status: completed",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == "Scenario file data/large.txt exceeds 2097152 bytes"
    else:
        raise AssertionError("Expected oversized repeated fixture to fail")


def test_run_scenario_reports_declared_file_parent_conflict(
    tmp_path: Path,
) -> None:
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

    try:
        run_scenario(
            load_scenario(scenario_path),
            RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
            ToolRegistry([ReadFileTool()]),
        )
    except ValueError as exc:
        assert str(exc) == (
            "Scenario file parent is not a directory: reports/summary.txt"
        )
    else:
        raise AssertionError("Expected conflicting scenario files to fail")


def test_run_scenario_reports_declared_file_parent_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "fixture-parent-stat.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: fixture-parent-stat",
                "goal: prepare fixture file",
                "files:",
                "  data/input.txt: fixture content",
                "actions:",
                "  - type: final",
                "    text: unreachable",
                "expect:",
                "  status: completed",
            ]
        ),
        encoding="utf-8",
    )
    original_exists = Path.exists

    def fail_parent_exists(path: Path) -> bool:
        if path.name == "data" and path.parent.parent.name == "workspaces":
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_parent_exists)

    try:
        run_scenario(
            load_scenario(scenario_path),
            RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
            ToolRegistry([ReadFileTool()]),
        )
    except ValueError as exc:
        assert str(exc) == (
            "Scenario file parent could not be checked: data/input.txt"
        )
    else:
        raise AssertionError("Expected scenario parent stat failure to fail")


def test_run_scenario_reports_declared_file_path_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "fixture-path-stat.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: fixture-path-stat",
                "goal: prepare fixture file",
                "files:",
                "  data/input.txt: fixture content",
                "actions:",
                "  - type: final",
                "    text: unreachable",
                "expect:",
                "  status: completed",
            ]
        ),
        encoding="utf-8",
    )
    original_exists = Path.exists

    def fail_target_exists(path: Path) -> bool:
        if path.name == "input.txt" and path.parent.name == "data":
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_target_exists)

    try:
        run_scenario(
            load_scenario(scenario_path),
            RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
            ToolRegistry([ReadFileTool()]),
        )
    except ValueError as exc:
        assert str(exc) == "Scenario file path could not be checked: data/input.txt"
    else:
        raise AssertionError("Expected scenario file stat failure to fail")


def test_run_scenario_reports_unencodable_declared_file_content(
    tmp_path: Path,
) -> None:
    scenario = eval_module.Scenario(
        name="fixture-unencodable",
        domain="filesystem",
        goal="prepare fixture",
        files={"data/input.txt": UnencodableString("fixture")},
        actions=[eval_module.FinalAction(text="unreachable")],
        expect={"status": "completed"},
    )

    try:
        run_scenario(
            scenario,
            RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
            ToolRegistry([ReadFileTool()]),
        )
    except ValueError as exc:
        assert str(exc) == (
            "Scenario file data/input.txt content could not be inspected"
        )
    else:
        raise AssertionError("Expected unencodable scenario file content to fail")


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


def test_eval_scenario_text_coercion_handles_unstringable_values() -> None:
    class UnstringableValue:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    value = UnstringableValue()

    assert eval_module._parse_file_content("data/input.txt", value) == (
        "<uninspectable>"
    )
    assert eval_module._parse_files({value: value}) == {
        "<uninspectable>": "<uninspectable>"
    }
    assert eval_module._parse_user_answers([value]) == ["<uninspectable>"]
    assert eval_module._parse_string_list([value], "Scenario contents") == [
        "<uninspectable>"
    ]

    conversation = eval_module._parse_conversation(
        [{"role": value, "content": value}]
    )
    assert conversation[0].role == "<uninspectable>"
    assert conversation[0].content == "<uninspectable>"

    final_action = eval_module._parse_action(
        {"type": "final", "text": value, "status": value},
        0,
    )
    assert final_action.text == "<uninspectable>"
    assert final_action.status == "<uninspectable>"


def test_load_scenario_rejects_uninspectable_goal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "uninspectable-goal.yaml"
    scenario_path.write_text("goal: finish\n", encoding="utf-8")

    monkeypatch.setattr(
        eval_module,
        "_load_yaml_document",
        lambda _path, _label: {
            "goal": UninspectableString("finish"),
            "actions": [{"type": "final", "text": "done"}],
            "expect": {"status": "completed"},
        },
    )

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == "Scenario goal could not be inspected"
    else:
        raise AssertionError("Expected uninspectable scenario goal to fail")


def test_eval_rejects_uninspectable_required_domains() -> None:
    try:
        eval_module._parse_required_domains(  # noqa: SLF001
            [UnstringableValue()]
        )
    except ValueError as exc:
        assert str(exc) == "Suite required_domains entries could not be inspected"
    else:
        raise AssertionError("Expected uninspectable required domain to fail")


def test_eval_normalizes_required_domain_text_subclasses() -> None:
    class StickyString(str):
        def __str__(self) -> str:
            return self

        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

    domains = eval_module._parse_required_domains(  # noqa: SLF001
        [StickyString("reliability")]
    )

    assert domains == ["reliability"]
    assert type(domains[0]) is str


def test_eval_rejects_uninspectable_tool_names() -> None:
    try:
        eval_module._parse_tool_call(  # noqa: SLF001
            {
                "tool": UninspectableString("read_file"),
                "arguments": {},
            },
            "Scenario action 1",
            1,
        )
    except ValueError as exc:
        assert str(exc) == "Scenario action 1 tool could not be inspected"
    else:
        raise AssertionError("Expected uninspectable tool name to fail")


def test_eval_normalizes_tool_name_text_subclasses() -> None:
    class StickyString(str):
        def __str__(self) -> str:
            return self

        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

    action = eval_module._parse_tool_call(  # noqa: SLF001
        {
            "tool": StickyString("read_file"),
            "arguments": {},
        },
        "Scenario action 1",
        1,
    )

    assert action.tool_name == "read_file"
    assert type(action.tool_name) is str


def test_eval_expectation_failures_handle_unreprable_values(tmp_path: Path) -> None:
    class UnreprableValue:
        def __repr__(self) -> str:
            raise RuntimeError("cannot repr")

    expected = UnreprableValue()
    actual = UnreprableValue()

    failures: list[str] = []
    eval_module._expect_equal(failures, "answer", {"answer": expected}, actual)
    assert failures == [
        "answer expected <uninspectable> but got <uninspectable>"
    ]

    failures = []
    eval_module._expect_observation_field(
        failures,
        1,
        "summary",
        {"summary": expected},
        actual,
    )
    assert failures == [
        "observation 1 summary expected <uninspectable> "
        "but got <uninspectable>"
    ]

    failures = []
    observation = SimpleNamespace(
        result=SimpleNamespace(payload={"answer": actual})
    )
    eval_module._expect_payload(failures, 1, {"answer": expected}, observation)
    assert failures == [
        "observation 1 payload.answer expected <uninspectable> "
        "but got <uninspectable>"
    ]

    failures = []
    log_path = tmp_path / "events.jsonl"
    log_path.write_text("", encoding="utf-8")
    eval_module._expect_events(failures, [{"payload": expected}], log_path)
    assert failures == [
        "event 1 expected {'payload': <uninspectable>} but was not found"
    ]


def test_eval_safe_repr_handles_uninspectable_collections() -> None:
    class ExplodingItems(dict):
        def items(self):  # type: ignore[override]
            raise RuntimeError("items unavailable")

    class ExplodingIter(list):
        def __iter__(self):  # type: ignore[override]
            raise RuntimeError("items unavailable")

    assert eval_module._safe_repr(ExplodingItems({"api_key": "secret"})) == (
        "<uninspectable>"
    )
    assert eval_module._safe_repr(ExplodingIter(["secret"])) == "<uninspectable>"


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
                "  - type: assert_conversation",
                "    contents:",
                "      - remember alpha",
                "      - alpha stored",
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


def test_run_scenario_reports_observation_call_id_mismatch(tmp_path: Path) -> None:
    scenario_path = tmp_path / "call-id-mismatch.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: call-id-mismatch",
                "goal: read a note",
                "files:",
                "  note.txt: hello",
                "actions:",
                "  - type: tool",
                "    tool: read_file",
                "    call_id: actual-call",
                "    arguments:",
                "      path: note.txt",
                "  - type: final",
                "    text: done",
                "expect:",
                "  status: completed",
                "  answer: done",
                "  observations: 1",
                "  observations_detail:",
                "    - call_id: expected-call",
                "      tool: read_file",
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
    assert result.failures == [
        "observation 1 call_id expected 'expected-call' but got 'actual-call'",
    ]


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


def test_eval_log_reader_reports_invalid_utf8_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    log_path.write_bytes(
        b'{"type": "run_started", "payload": {"goal": "finish"}}\n'
        b"\xff\xfe\x00broken\n"
        b'{"type": "run_finished", "payload": {"status": "completed"}}\n'
    )
    failures: list[str] = []

    events = _read_log_events(log_path, failures)

    assert [event["type"] for event in events] == ["run_started", "run_finished"]
    assert failures == ["log line 2 could not be decoded as UTF-8"]


def test_eval_log_reader_reports_too_deep_json_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    too_deep = '{"child":' * 20_000 + "null" + "}" * 20_000
    log_path.write_text(
        "\n".join(
            [
                '{"type": "run_started", "payload": {"goal": "finish"}}',
                too_deep,
                '{"type": "run_finished", "payload": {"status": "completed"}}',
            ]
        ),
        encoding="utf-8",
    )
    failures: list[str] = []

    events = _read_log_events(log_path, failures)

    assert [event["type"] for event in events] == ["run_started", "run_finished"]
    assert failures == ["log line 2 could not be decoded: maximum nesting depth"]


def test_eval_log_reader_reports_log_path_directory(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    log_path.mkdir()
    failures: list[str] = []

    events = _read_log_events(log_path, failures)

    assert events == []
    assert failures == [f"log file expected but was not a file: {log_path}"]


def test_eval_log_reader_reports_log_exists_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "run.jsonl"
    original_exists = Path.exists

    def fail_log_exists(path: Path) -> bool:
        if path == log_path:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_log_exists)
    failures: list[str] = []

    events = _read_log_events(log_path, failures)

    assert events == []
    assert failures == [f"log file could not be checked: {log_path}"]


def test_eval_log_reader_reports_log_file_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "run.jsonl"
    log_path.write_text('{"type": "run_started"}\n', encoding="utf-8")
    original_is_file = Path.is_file

    def fail_log_is_file(path: Path) -> bool:
        if path == log_path:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_log_is_file)
    failures: list[str] = []

    events = _read_log_events(log_path, failures)

    assert events == []
    assert failures == [f"log file could not be checked: {log_path}"]


def test_eval_log_reader_reports_log_read_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "run.jsonl"
    log_path.write_text('{"type": "run_started"}\n', encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def fail_read_bytes(path: Path) -> bytes:
        if path == log_path:
            raise OSError("permission denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    failures: list[str] = []

    events = _read_log_events(log_path, failures)

    assert events == []
    assert failures == [f"log file could not be read: {log_path}"]


def test_eval_log_reader_rejects_oversized_log_before_reading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "run.jsonl"
    log_path.write_bytes(b"x" * 11)
    monkeypatch.setattr(eval_module, "MAX_SCENARIO_FILE_BYTES", 10)
    original_read_bytes = Path.read_bytes

    def fail_read_bytes(path: Path) -> bytes:
        if path == log_path:
            raise AssertionError("oversized log file should not be read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    failures: list[str] = []

    events = _read_log_events(log_path, failures)

    assert events == []
    assert failures == [f"log file exceeds 10 bytes: {log_path}"]


def test_eval_log_reader_rejects_log_that_grows_after_stat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "run.jsonl"
    log_path.write_bytes(b"{}")
    monkeypatch.setattr(eval_module, "MAX_SCENARIO_FILE_BYTES", 10)
    original_read_bytes = Path.read_bytes

    def grow_read_bytes(path: Path) -> bytes:
        if path == log_path:
            return b"x" * 11
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", grow_read_bytes)
    failures: list[str] = []

    events = _read_log_events(log_path, failures)

    assert events == []
    assert failures == [f"log file exceeds 10 bytes: {log_path}"]


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


def test_run_scenario_reports_invalid_utf8_expected_file(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "summary.txt"
    report_path.parent.mkdir()
    report_path.write_bytes(b"\xff\xfe\x00broken")
    scenario_path = tmp_path / "binary-report.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: binary-report",
                "goal: inspect report",
                "actions:",
                "  - type: final",
                "    text: checked",
                "expect:",
                "  status: completed",
                "  files:",
                "    reports/summary.txt: expected text",
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
        "file reports/summary.txt could not be decoded as UTF-8"
    ]


def test_run_scenario_reports_expected_file_read_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "reports" / "summary.txt"
    report_path.parent.mkdir()
    report_path.write_text("summary", encoding="utf-8")
    scenario_path = tmp_path / "unreadable-report.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: unreadable-report",
                "goal: inspect report",
                "actions:",
                "  - type: final",
                "    text: checked",
                "expect:",
                "  status: completed",
                "  files:",
                "    reports/summary.txt: summary",
            ]
        ),
        encoding="utf-8",
    )
    scenario = load_scenario(scenario_path)
    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs) -> str:
        if path == report_path:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    result = run_scenario(
        scenario,
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is False
    assert result.failures == ["file reports/summary.txt could not be read"]


def test_run_scenario_rejects_oversized_expected_file_before_reading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "reports" / "summary.txt"
    report_path.parent.mkdir()
    report_path.write_text("x" * 11, encoding="utf-8")
    scenario_path = tmp_path / "large-report.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: large-report",
                "goal: inspect report",
                "actions:",
                "  - type: final",
                "    text: checked",
                "expect:",
                "  status: completed",
                "  files:",
                "    reports/summary.txt: summary",
            ]
        ),
        encoding="utf-8",
    )
    scenario = load_scenario(scenario_path)
    monkeypatch.setattr(eval_module, "MAX_SCENARIO_FILE_BYTES", 10)
    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs) -> str:
        if path == report_path:
            raise AssertionError("oversized expected file should not be read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    result = run_scenario(
        scenario,
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is False
    assert result.failures == ["file reports/summary.txt exceeds 10 bytes"]


def test_run_scenario_rejects_expected_file_that_grows_after_stat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "reports" / "summary.txt"
    report_path.parent.mkdir()
    report_path.write_text("x", encoding="utf-8")
    scenario_path = tmp_path / "large-report-after-stat.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: large-report-after-stat",
                "goal: inspect report",
                "actions:",
                "  - type: final",
                "    text: checked",
                "expect:",
                "  status: completed",
                "  files:",
                "    reports/summary.txt: summary",
            ]
        ),
        encoding="utf-8",
    )
    scenario = load_scenario(scenario_path)
    monkeypatch.setattr(eval_module, "MAX_SCENARIO_FILE_BYTES", 10)
    original_read_text = Path.read_text

    def grow_read_text(path: Path, *args, **kwargs) -> str:
        if path == report_path:
            return "x" * 11
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", grow_read_text)

    result = run_scenario(
        scenario,
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is False
    assert result.failures == ["file reports/summary.txt exceeds 10 bytes"]


def test_run_scenario_reports_expected_file_exists_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "reports" / "summary.txt"
    report_path.parent.mkdir()
    report_path.write_text("summary", encoding="utf-8")
    scenario_path = tmp_path / "report-stat.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: report-stat",
                "goal: inspect report",
                "actions:",
                "  - type: final",
                "    text: checked",
                "expect:",
                "  status: completed",
                "  files:",
                "    reports/summary.txt: summary",
            ]
        ),
        encoding="utf-8",
    )
    scenario = load_scenario(scenario_path)
    original_exists = Path.exists

    def fail_report_exists(path: Path) -> bool:
        if path == report_path:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_report_exists)

    result = run_scenario(
        scenario,
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is False
    assert result.failures == ["file reports/summary.txt could not be checked"]


def test_run_scenario_reports_expected_file_type_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "reports" / "summary.txt"
    report_path.parent.mkdir()
    report_path.write_text("summary", encoding="utf-8")
    scenario_path = tmp_path / "report-type-stat.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: report-type-stat",
                "goal: inspect report",
                "actions:",
                "  - type: final",
                "    text: checked",
                "expect:",
                "  status: completed",
                "  files:",
                "    reports/summary.txt: summary",
            ]
        ),
        encoding="utf-8",
    )
    scenario = load_scenario(scenario_path)
    original_is_file = Path.is_file

    def fail_report_is_file(path: Path) -> bool:
        if path == report_path:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_report_is_file)

    result = run_scenario(
        scenario,
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is False
    assert result.failures == ["file reports/summary.txt could not be checked"]


def test_run_scenario_reports_unencodable_expected_file_content(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "reports" / "summary.txt"
    report_path.parent.mkdir()
    report_path.write_text("summary", encoding="utf-8")
    scenario_path = tmp_path / "report-unencodable.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: report-unencodable",
                "goal: inspect report",
                "actions:",
                "  - type: final",
                "    text: checked",
                "expect:",
                "  status: completed",
                "  files:",
                "    reports/summary.txt: summary",
            ]
        ),
        encoding="utf-8",
    )
    scenario = load_scenario(scenario_path)
    original_read_text = Path.read_text

    def unencodable_report_text(path: Path, *args, **kwargs) -> str:
        if path == report_path:
            return UnencodableString("summary")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unencodable_report_text)

    result = run_scenario(
        scenario,
        RunConfig(workspace=tmp_path, logs_dir=tmp_path / "runs"),
        ToolRegistry([ReadFileTool()]),
    )

    assert result.passed is False
    assert result.failures == ["file reports/summary.txt could not be inspected"]


def test_load_scenario_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    try:
        load_scenario(missing_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file not found: {missing_path}"
    else:
        raise AssertionError("Expected missing scenario file to fail")


def test_load_scenario_reports_exists_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "stat.yaml"
    original_exists = Path.exists

    def fail_scenario_exists(path: Path) -> bool:
        if path == scenario_path:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_scenario_exists)

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file could not be checked: {scenario_path}"
    else:
        raise AssertionError("Expected scenario stat failure to fail")


def test_load_scenario_rejects_directory_path(tmp_path: Path) -> None:
    scenario_path = tmp_path / "directory.yaml"
    scenario_path.mkdir()

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file must be a file: {scenario_path}"
    else:
        raise AssertionError("Expected scenario directory path to fail")


def test_load_scenario_reports_file_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "stat.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "goal: finish",
                "actions:",
                "  - type: final",
                "    text: done",
                "expect:",
                "  status: completed",
            ]
        ),
        encoding="utf-8",
    )
    original_is_file = Path.is_file

    def fail_scenario_is_file(path: Path) -> bool:
        if path == scenario_path:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_scenario_is_file)

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file could not be checked: {scenario_path}"
    else:
        raise AssertionError("Expected scenario file stat failure to fail")


def test_load_scenario_reports_malformed_yaml(tmp_path: Path) -> None:
    scenario_path = tmp_path / "broken.yaml"
    scenario_path.write_text("goal: [unterminated\n", encoding="utf-8")

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file could not be parsed: {scenario_path}"
    else:
        raise AssertionError("Expected malformed scenario to fail")


def test_load_scenario_reports_deeply_nested_yaml(tmp_path: Path) -> None:
    scenario_path = tmp_path / "deep.yaml"
    scenario_path.write_text("[" * 20_000 + "null" + "]" * 20_000, encoding="utf-8")

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file could not be parsed: {scenario_path}"
    else:
        raise AssertionError("Expected deeply nested scenario to fail")


def test_load_scenario_reports_invalid_utf8_yaml(tmp_path: Path) -> None:
    scenario_path = tmp_path / "broken.yaml"
    scenario_path.write_bytes(b"\xff\xfe\x00broken")

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file could not be read as UTF-8: {scenario_path}"
    else:
        raise AssertionError("Expected invalid UTF-8 scenario to fail")


def test_load_scenario_reports_yaml_read_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "unreadable.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "goal: finish",
                "actions:",
                "  - type: final",
                "    text: done",
                "expect:",
                "  status: completed",
            ]
        ),
        encoding="utf-8",
    )
    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs) -> str:
        if path == scenario_path:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file could not be read: {scenario_path}"
    else:
        raise AssertionError("Expected unreadable scenario to fail")


def test_load_scenario_rejects_oversized_yaml_before_reading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "large.yaml"
    scenario_path.write_text("x" * 11, encoding="utf-8")
    monkeypatch.setattr(eval_module, "MAX_SCENARIO_FILE_BYTES", 10)
    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs) -> str:
        if path == scenario_path:
            raise AssertionError("oversized scenario file should not be read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file exceeds 10 bytes: {scenario_path}"
    else:
        raise AssertionError("Expected oversized scenario to fail")


def test_load_scenario_rejects_yaml_that_grows_after_stat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "large-after-stat.yaml"
    scenario_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(eval_module, "MAX_SCENARIO_FILE_BYTES", 10)
    original_read_text = Path.read_text

    def grow_read_text(path: Path, *args, **kwargs) -> str:
        if path == scenario_path:
            return "x" * 11
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", grow_read_text)

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file exceeds 10 bytes: {scenario_path}"
    else:
        raise AssertionError("Expected scenario file growth to fail")


def test_load_scenario_reports_unencodable_yaml_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "unencodable.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "goal: finish",
                "actions:",
                "  - type: final",
                "    text: done",
                "expect:",
                "  status: completed",
            ]
        ),
        encoding="utf-8",
    )
    original_read_text = Path.read_text

    def unencodable_read_text(path: Path, *args, **kwargs) -> str:
        if path == scenario_path:
            return UnencodableString("goal: finish\n")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unencodable_read_text)

    try:
        load_scenario(scenario_path)
    except ValueError as exc:
        assert str(exc) == f"Scenario file could not be inspected: {scenario_path}"
    else:
        raise AssertionError("Expected unencodable scenario text to fail")


def test_load_scenario_rejects_unencodable_repeated_workspace_file() -> None:
    try:
        eval_module._parse_file_content(  # noqa: SLF001
            "data/input.txt",
            {"repeat": UnencodableString("x"), "count": 1},
        )
    except ValueError as exc:
        assert str(exc) == (
            "Scenario file data/input.txt repeat could not be inspected"
        )
    else:
        raise AssertionError("Expected unencodable repeated fixture to fail")


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


def test_load_scenario_suite_reports_directory_read_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    original_rglob = Path.rglob

    def fail_rglob(path: Path, pattern: str):
        if path == suite_dir:
            raise OSError("permission denied")
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    try:
        load_scenario_suite(suite_dir)
    except ValueError as exc:
        assert str(exc) == f"Scenario path could not be read: {suite_dir}"
    else:
        raise AssertionError("Expected unreadable scenario suite to fail")


def test_load_scenario_suite_reports_exists_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_dir = tmp_path / "evals"
    original_exists = Path.exists

    def fail_suite_exists(path: Path) -> bool:
        if path == suite_dir:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_suite_exists)

    try:
        load_scenario_suite(suite_dir)
    except ValueError as exc:
        assert str(exc) == f"Scenario path could not be checked: {suite_dir}"
    else:
        raise AssertionError("Expected scenario suite stat failure to fail")


def test_load_scenario_suite_reports_file_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    original_is_file = Path.is_file

    def fail_suite_is_file(path: Path) -> bool:
        if path == suite_dir:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_suite_is_file)

    try:
        load_scenario_suite(suite_dir)
    except ValueError as exc:
        assert str(exc) == f"Scenario path could not be checked: {suite_dir}"
    else:
        raise AssertionError("Expected scenario suite file stat failure to fail")


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


def test_load_scenario_suite_config_reports_malformed_manifest(
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    manifest = suite_dir / "suite.yaml"
    manifest.write_text("required_domains: [unterminated\n", encoding="utf-8")

    try:
        load_scenario_suite_config(suite_dir)
    except ValueError as exc:
        assert str(exc) == f"Suite manifest could not be parsed: {manifest}"
    else:
        raise AssertionError("Expected malformed suite manifest to fail")


def test_load_scenario_suite_config_reports_invalid_utf8_manifest(
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    manifest = suite_dir / "suite.yaml"
    manifest.write_bytes(b"\xff\xfe\x00broken")

    try:
        load_scenario_suite_config(suite_dir)
    except ValueError as exc:
        assert str(exc) == f"Suite manifest could not be read as UTF-8: {manifest}"
    else:
        raise AssertionError("Expected invalid UTF-8 suite manifest to fail")


def test_load_scenario_suite_config_reports_manifest_read_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    manifest = suite_dir / "suite.yaml"
    manifest.write_text(
        "\n".join(
            [
                "required_domains:",
                "  - finance",
            ]
        ),
        encoding="utf-8",
    )
    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs) -> str:
        if path == manifest:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    try:
        load_scenario_suite_config(suite_dir)
    except ValueError as exc:
        assert str(exc) == f"Suite manifest could not be read: {manifest}"
    else:
        raise AssertionError("Expected unreadable suite manifest to fail")


def test_load_scenario_suite_config_reports_manifest_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    manifest = suite_dir / "suite.yaml"
    original_exists = Path.exists

    def fail_manifest_exists(path: Path) -> bool:
        if path == manifest:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_manifest_exists)

    try:
        load_scenario_suite_config(suite_dir)
    except ValueError as exc:
        assert str(exc) == f"Suite manifest path could not be checked: {suite_dir}"
    else:
        raise AssertionError("Expected suite manifest stat failure to fail")


def test_load_scenario_suite_config_rejects_manifest_directory(
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    manifest = suite_dir / "suite.yaml"
    manifest.mkdir()

    try:
        load_scenario_suite_config(suite_dir)
    except ValueError as exc:
        assert str(exc) == f"Suite manifest must be a file: {manifest}"
    else:
        raise AssertionError("Expected suite manifest directory to fail")


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
