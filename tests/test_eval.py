from pathlib import Path

from agentskeleton.config import RunConfig
from agentskeleton.eval import load_scenario, run_scenario
from agentskeleton.tools.filesystem import ReadFileTool
from agentskeleton.tools.registry import ToolRegistry


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
