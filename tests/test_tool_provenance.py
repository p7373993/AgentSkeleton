import hashlib
import json

from agentskeleton.tools.base import Tool, ToolContext, ToolResult
from agentskeleton.tools.provenance import (
    registered_tool_inventory,
    tool_schema_hash,
)
from agentskeleton.tools.registry import ToolRegistry


class ProvenanceTool(Tool):
    name = "provenance"
    description = "Return provenance."
    risk = "read"
    args_schema = {
        "required": ["value"],
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "string"}},
    }

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(success=True, payload={}, summary="ok")


def expected_schema_hash(schema: object) -> str:
    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_tool_schema_hash_is_stable_for_json_key_order() -> None:
    first = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    second = {
        "required": ["value"],
        "additionalProperties": False,
        "properties": {"value": {"type": "string"}},
        "type": "object",
    }

    assert tool_schema_hash(first) == expected_schema_hash(first)
    assert tool_schema_hash(first) == tool_schema_hash(second)


def test_registered_tool_inventory_reports_deterministic_provenance() -> None:
    registry = ToolRegistry([ProvenanceTool()])

    assert registered_tool_inventory(registry) == [
        {
            "name": "provenance",
            "description": "Return provenance.",
            "risk": "read",
            "args_schema_hash": expected_schema_hash(ProvenanceTool.args_schema),
            "implementation": "test_tool_provenance.ProvenanceTool",
        }
    ]
