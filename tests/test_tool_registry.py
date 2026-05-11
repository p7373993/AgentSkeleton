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


def test_registry_rejects_non_tool_values() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Registered value must be a Tool"):
        registry.register(object())


def test_registry_rejects_non_string_tool_names() -> None:
    class NonStringNameTool(EchoTool):
        name = 123

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool name must be a string"):
        registry.register(NonStringNameTool())


def test_registry_rejects_empty_tool_names() -> None:
    class EmptyNameTool(EchoTool):
        name = " "

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool name cannot be empty"):
        registry.register(EmptyNameTool())


def test_registry_rejects_tool_names_with_surrounding_whitespace() -> None:
    class WhitespaceNameTool(EchoTool):
        name = " echo "

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool name cannot contain whitespace"):
        registry.register(WhitespaceNameTool())


def test_registry_rejects_tool_names_with_invalid_characters() -> None:
    class InvalidNameTool(EchoTool):
        name = "bad name!"

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool name must match"):
        registry.register(InvalidNameTool())


def test_registry_rejects_non_string_tool_descriptions() -> None:
    class InvalidDescriptionTool(EchoTool):
        description = 123

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool echo description must be a string"):
        registry.register(InvalidDescriptionTool())


@pytest.mark.parametrize(
    ("risk", "error"),
    [
        (123, "Tool echo risk must be a string"),
        (" ", "Tool echo risk cannot be empty"),
        (" read ", "Tool echo risk cannot contain whitespace"),
    ],
)
def test_registry_rejects_invalid_tool_risk_metadata(
    risk: object,
    error: str,
) -> None:
    class InvalidRiskTool(EchoTool):
        pass

    InvalidRiskTool.risk = risk
    registry = ToolRegistry()

    with pytest.raises(ValueError, match=error):
        registry.register(InvalidRiskTool())


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        ([], "schema must be a mapping"),
        ({"type": "array", "items": {"type": "string"}}, "schema type must be object"),
        ({"type": "object", "properties": []}, "schema properties must be a mapping"),
    ],
)
def test_registry_rejects_invalid_args_schema_shape(
    schema: object,
    error: str,
) -> None:
    class InvalidSchemaTool(EchoTool):
        args_schema = schema

    registry = ToolRegistry()

    with pytest.raises(ValueError, match=error):
        registry.register(InvalidSchemaTool())


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        (
            {"type": "object", "properties": {"text": []}},
            "schema property text must be a mapping",
        ),
        (
            {"type": "object", "properties": {1: {"type": "string"}}},
            "schema property names must be strings",
        ),
        (
            {"type": "object", "properties": {}, "required": "text"},
            "schema required must be a list",
        ),
        (
            {"type": "object", "properties": {}, "required": [1]},
            "schema required entries must be strings",
        ),
    ],
)
def test_registry_rejects_invalid_args_schema_members(
    schema: object,
    error: str,
) -> None:
    class InvalidSchemaTool(EchoTool):
        args_schema = schema

    registry = ToolRegistry()

    with pytest.raises(ValueError, match=error):
        registry.register(InvalidSchemaTool())


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        (
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": 123,
                    }
                },
            },
            "schema property text type must be a string or list of strings",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": ["string", 123],
                    }
                },
            },
            "schema property text type entries must be strings",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "settings": {
                        "type": "object",
                        "properties": [],
                    }
                },
            },
            "schema property settings properties must be a mapping",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "settings": {
                        "type": "object",
                        "properties": {1: {"type": "string"}},
                    }
                },
            },
            "schema property settings property names must be strings",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "settings": {
                        "type": "object",
                        "properties": {},
                        "required": "enabled",
                    }
                },
            },
            "schema property settings required must be a list",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": [],
                    }
                },
            },
            "schema property tags items must be a mapping",
        ),
    ],
)
def test_registry_rejects_invalid_nested_args_schema_members(
    schema: object,
    error: str,
) -> None:
    class InvalidSchemaTool(EchoTool):
        args_schema = schema

    registry = ToolRegistry()

    with pytest.raises(ValueError, match=error):
        registry.register(InvalidSchemaTool())


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


def test_registry_exports_openai_function_schemas_as_isolated_copies() -> None:
    registry = ToolRegistry([EchoTool()])

    schemas = registry.to_openai_tools()
    schemas[0]["parameters"]["required"].append("extra")
    schemas[0]["parameters"]["properties"]["text"]["type"] = "integer"

    assert EchoTool.args_schema == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    assert registry.to_openai_tools()[0]["parameters"] == EchoTool.args_schema


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


def test_load_tools_from_module_reimports_current_module_path(
    tmp_path,
    monkeypatch,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    _write_tool_module(first_dir / "custom_tools.py", "first_tool")
    _write_tool_module(second_dir / "custom_tools.py", "second_tool")

    monkeypatch.syspath_prepend(str(first_dir))
    assert [tool.name for tool in load_tools_from_modules(["custom_tools"])] == [
        "first_tool"
    ]

    monkeypatch.syspath_prepend(str(second_dir))
    assert [tool.name for tool in load_tools_from_modules(["custom_tools"])] == [
        "second_tool"
    ]


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


def test_load_tools_from_module_rejects_non_callable_register_tools(
    tmp_path,
    monkeypatch,
) -> None:
    module_path = tmp_path / "bad_register_tools.py"
    module_path.write_text("register_tools = []\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(
        ValueError,
        match="Tool module bad_register_tools register_tools must be callable",
    ):
        load_tools_from_modules(["bad_register_tools"])


def test_load_tools_from_module_reports_register_tools_failures(
    tmp_path,
    monkeypatch,
) -> None:
    module_path = tmp_path / "failing_register_tools.py"
    module_path.write_text(
        "\n".join(
            [
                "def register_tools(registry):",
                "    raise RuntimeError('boom')",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(
        ValueError,
        match="Tool module failing_register_tools register_tools failed: RuntimeError",
    ):
        load_tools_from_modules(["failing_register_tools"])


def test_load_tools_from_module_requires_tool_provider(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "empty_tools.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="must define TOOLS or register_tools"):
        load_tools_from_modules(["empty_tools"])


def test_load_tools_from_module_reports_missing_module() -> None:
    with pytest.raises(ValueError, match="Tool module not found: missing_tools"):
        load_tools_from_modules(["missing_tools"])


def test_load_tools_from_module_reports_import_failures(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "bad_syntax_tools.py"
    module_path.write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(
        ValueError,
        match="Tool module bad_syntax_tools could not be imported",
    ):
        load_tools_from_modules(["bad_syntax_tools"])


def _write_tool_module(path, tool_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "from agentskeleton.tools.base import Tool, ToolResult",
                "",
                "class CustomTool(Tool):",
                f"    name = '{tool_name}'",
                "    description = 'Custom tool.'",
                "    risk = 'read'",
                "    args_schema = {'type': 'object', 'properties': {}}",
                "    def execute(self, args, context):",
                "        return ToolResult(success=True, summary='ok')",
                "",
                "TOOLS = [CustomTool()]",
            ]
        ),
        encoding="utf-8",
    )
