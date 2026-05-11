from typing import Any, ClassVar

from agentskeleton.tools.base import Tool, ToolContext, ToolResult


class ExplodingTool(Tool):
    name: ClassVar[str] = "explode"
    description: ClassVar[str] = "Raise a deterministic exception for evals."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        raise RuntimeError("boom")


class InvalidResultTool(Tool):
    name: ClassVar[str] = "invalid_result"
    description: ClassVar[str] = "Return a non-ToolResult value for evals."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        return {"success": True, "summary": "not a model"}  # type: ignore[return-value]


TOOLS = [ExplodingTool(), InvalidResultTool()]
