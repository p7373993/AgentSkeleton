from pathlib import Path

from agentskeleton.config import RunConfig
from agentskeleton.eval import (
    load_scenario,
    load_scenario_suite,
    run_scenario,
    run_scenario_suite,
)
from agentskeleton.tools.filesystem import ReadFileTool, WriteFileTool
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


def test_run_scenario_suite_summarizes_passes_and_failures(tmp_path: Path) -> None:
    passing = load_scenario(_write_final_scenario(tmp_path, "passing", "ok", "ok"))
    failing = load_scenario(
        _write_final_scenario(tmp_path, "failing", "actual", "expected")
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
    assert [result.scenario for result in suite.results] == ["passing", "failing"]
    assert suite.results[1].failures == [
        "answer expected 'expected' but got 'actual'"
    ]


def _write_final_scenario(
    tmp_path: Path,
    name: str,
    final_text: str,
    expected_answer: str,
) -> Path:
    scenario_path = tmp_path / f"{name}.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                f"name: {name}",
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
