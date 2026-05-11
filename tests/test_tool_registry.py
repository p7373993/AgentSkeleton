import pytest

from agentskeleton.tools.base import Tool, ToolContext, ToolResult
from agentskeleton.tools.loading import load_tools_from_modules
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


def test_load_tools_from_module_tools_list(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "custom_tools.py"
    module_path.write_text(
        "\n".join(
            [
                "from agentskeleton.tools.base import Tool, ToolResult",
                "",
                "class CustomTool(Tool):",
                "    name = 'custom_echo'",
                "    description = 'Custom echo.'",
                "    risk = 'read'",
                "    args_schema = {",
                "        'type': 'object',",
                "        'properties': {'text': {'type': 'string'}},",
                "        'required': ['text'],",
                "        'additionalProperties': False,",
                "    }",
                "    def execute(self, args, context):",
                "        return ToolResult(",
                "            success=True,",
                "            payload={'text': args['text']},",
                "            summary=args['text'],",
                "        )",
                "",
                "TOOLS = [CustomTool()]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    tools = load_tools_from_modules(["custom_tools"])

    assert [tool.name for tool in tools] == ["custom_echo"]


def test_load_tools_from_module_register_function(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "custom_register_tools.py"
    module_path.write_text(
        "\n".join(
            [
                "from agentskeleton.tools.base import Tool, ToolResult",
                "",
                "class CustomTool(Tool):",
                "    name = 'registered_echo'",
                "    description = 'Registered echo.'",
                "    risk = 'read'",
                "    args_schema = {'type': 'object', 'properties': {}}",
                "    def execute(self, args, context):",
                "        return ToolResult(success=True, summary='ok')",
                "",
                "def register_tools(registry):",
                "    registry.register(CustomTool())",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    tools = load_tools_from_modules(["custom_register_tools"])

    assert [tool.name for tool in tools] == ["registered_echo"]


def test_load_tools_from_module_requires_tool_provider(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "empty_tools.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="must define TOOLS or register_tools"):
        load_tools_from_modules(["empty_tools"])


def test_load_tools_from_module_reports_missing_module() -> None:
    with pytest.raises(ValueError, match="Tool module not found: missing_tools"):
        load_tools_from_modules(["missing_tools"])
