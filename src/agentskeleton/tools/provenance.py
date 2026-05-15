import hashlib
import json

from agentskeleton.tools.base import Tool
from agentskeleton.tools.registry import ToolRegistry


def tool_schema_hash(schema: object) -> str:
    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tool_implementation(tool: object) -> str:
    tool_type = type(tool)
    return f"{tool_type.__module__}.{tool_type.__qualname__}"


def registered_tool_provenance(tool: Tool) -> dict[str, str]:
    return {
        "name": tool.name,
        "description": tool.description,
        "risk": tool.risk,
        "args_schema_hash": tool_schema_hash(tool.args_schema),
        "implementation": tool_implementation(tool),
    }


def registered_tool_inventory(registry: ToolRegistry) -> list[dict[str, str]]:
    return [registered_tool_provenance(tool) for tool in registry.all()]
