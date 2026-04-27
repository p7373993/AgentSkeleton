import pytest

from agentskeleton.tools.base import Tool, ToolContext, ToolResult
from agentskeleton.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Echo text."
    risk = "read"
    args_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            payload={"text": args["text"], "workspace": str(context.workspace)},
            summary=str(args["text"]),
        )


def test_registry_registers_and_executes_tool(tmp_path) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    tool = registry.get("echo")
    result = tool.execute({"text": "hello"}, ToolContext(workspace=tmp_path))

    assert result.success is True
    assert result.payload["text"] == "hello"
    assert result.summary == "hello"


def test_registry_rejects_duplicate_tools() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())


def test_registry_exports_openai_function_schemas() -> None:
    registry = ToolRegistry([EchoTool()])

    schemas = registry.to_openai_tools()

    assert schemas == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo text.",
            "parameters": EchoTool.args_schema,
        }
    ]


def test_registry_raises_for_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="Unknown tool"):
        registry.get("missing")
