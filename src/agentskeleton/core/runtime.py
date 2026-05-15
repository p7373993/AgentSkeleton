from agentskeleton.config import RunConfig
from agentskeleton.tools.provenance import registered_tool_inventory
from agentskeleton.tools.registry import ToolRegistry


def runtime_metadata(config: RunConfig, registry: ToolRegistry) -> dict[str, object]:
    enabled_tools = (
        None if config.enabled_tools is None else list(config.enabled_tools)
    )
    return {
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "text_verbosity": config.text_verbosity,
        "max_steps": config.max_steps,
        "model_retry_attempts": config.model_retry_attempts,
        "permission_profile": config.permission_profile,
        "confirm_risky_actions": config.confirm_risky_actions,
        "enabled_tools": enabled_tools,
        "tool_modules": list(config.tool_modules),
        "registered_tools": registered_tool_inventory(registry),
    }
