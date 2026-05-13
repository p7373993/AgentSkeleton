from pathlib import Path

from agentskeleton.cli import build_default_registry
from agentskeleton.config import RunConfig
from agentskeleton.eval import (
    load_scenario_suite,
    load_scenario_suite_config,
    run_scenario_suite,
)


def test_checked_in_eval_suite_passes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_config = load_scenario_suite_config(repo_root / "evals")
    scenarios = load_scenario_suite(repo_root / "evals")
    result = run_scenario_suite(
        scenarios,
        RunConfig(workspace=repo_root, logs_dir=tmp_path / "runs"),
        registry_factory=lambda config: build_default_registry(
            config.enabled_tools,
            config.tool_modules,
        ),
        required_domains=suite_config.required_domains,
    )

    assert suite_config.required_domains == [
        "artifacts",
        "coding",
        "data",
        "filesystem",
        "interactive",
        "reliability",
        "tool_packs",
        "writing",
    ]
    assert result.passed is True
    assert result.total == 47
    assert result.failed_count == 0
    assert "coding-edit-test" in {item.scenario for item in result.results}
    assert "data-csv-report-artifact" in {item.scenario for item in result.results}
    assert "interactive-answer-artifact" in {
        item.scenario for item in result.results
    }
    assert "invalid-tool-result" in {item.scenario for item in result.results}
    assert result.domains == {
        "artifacts": {"passed": 1, "failed": 0, "total": 1},
        "coding": {"passed": 2, "failed": 0, "total": 2},
        "data": {"passed": 2, "failed": 0, "total": 2},
        "filesystem": {"passed": 4, "failed": 0, "total": 4},
        "interactive": {"passed": 2, "failed": 0, "total": 2},
        "reliability": {"passed": 34, "failed": 0, "total": 34},
        "tool_packs": {"passed": 1, "failed": 0, "total": 1},
        "writing": {"passed": 1, "failed": 0, "total": 1},
    }
