from pathlib import Path

from agentskeleton.config import RunConfig
from agentskeleton.core.runtime import runtime_metadata
from agentskeleton.tools.filesystem import ReadFileTool
from agentskeleton.tools.provenance import registered_tool_inventory
from agentskeleton.tools.registry import ToolRegistry


def test_runtime_metadata_reports_config_and_registered_tools(tmp_path: Path) -> None:
    config = RunConfig(
        workspace=tmp_path,
        model="gpt-5.4-mini",
        reasoning_effort="medium",
        text_verbosity="low",
        max_steps=7,
        model_retry_attempts=3,
        permission_profile="trusted",
        confirm_risky_actions=False,
        enabled_tools=["read_file"],
        tool_modules=["example_tools"],
    )
    registry = ToolRegistry([ReadFileTool()])

    assert runtime_metadata(config, registry) == {
        "model": "gpt-5.4-mini",
        "reasoning_effort": "medium",
        "text_verbosity": "low",
        "max_steps": 7,
        "model_retry_attempts": 3,
        "permission_profile": "trusted",
        "confirm_risky_actions": False,
        "enabled_tools": ["read_file"],
        "tool_modules": ["example_tools"],
        "registered_tools": registered_tool_inventory(registry),
    }
