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


TOOLS = [ExplodingTool()]
