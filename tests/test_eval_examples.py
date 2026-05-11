from pathlib import Path

from agentskeleton.config import RunConfig
from agentskeleton.eval import load_scenario_suite, run_scenario_suite
from agentskeleton.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from agentskeleton.tools.registry import ToolRegistry


def test_checked_in_eval_suite_passes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scenarios = load_scenario_suite(repo_root / "evals")
    result = run_scenario_suite(
        scenarios,
        RunConfig(workspace=repo_root, logs_dir=tmp_path / "runs"),
        ToolRegistry([ListDirTool(), ReadFileTool(), WriteFileTool()]),
    )

    assert result.passed is True
    assert result.total == 6
    assert result.failed_count == 0
