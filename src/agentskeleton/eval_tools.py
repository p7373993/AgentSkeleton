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


class ObjectEchoTool(Tool):
    name: ClassVar[str] = "object_echo"
    description: ClassVar[str] = "Echo an object payload for evals."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"payload": {"type": "object"}},
        "required": ["payload"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            payload={"payload": args["payload"]},
            summary="object echo ok",
        )


TOOLS = [ExplodingTool(), InvalidResultTool(), ObjectEchoTool()]
