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


class StickyString(str):
    def __str__(self) -> str:
        return self

    def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self


def test_registry_registers_and_executes_tool(tmp_path) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    tool = registry.get("echo")
    result = tool.execute({"text": "hello"}, ToolContext(workspace=tmp_path))

    assert result.success is True
    assert result.payload["text"] == "hello"
    assert result.summary == "hello"


def test_registry_initializes_from_iterable_without_length() -> None:
    class ExplodingToolList(list):
        def __len__(self) -> int:
            raise RuntimeError("tool count unavailable")

    registry = ToolRegistry(ExplodingToolList([EchoTool()]))

    assert registry.get("echo").name == "echo"


def test_registry_rejects_duplicate_tools() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())


def test_registry_rejects_too_many_tools() -> None:
    registry = ToolRegistry()

    for index in range(128):
        tool_type = type(
            f"EchoTool{index}",
            (EchoTool,),
            {"name": f"echo_{index}"},
        )
        registry.register(tool_type())

    class ExtraTool(EchoTool):
        name = "echo_extra"

    with pytest.raises(ValueError, match="Too many tools registered"):
        registry.register(ExtraTool())


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


def test_registry_rejects_oversized_tool_names() -> None:
    class OversizedNameTool(EchoTool):
        name = "t" * 513

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool name exceeds maximum size"):
        registry.register(OversizedNameTool())


def test_registry_rejects_unencodable_tool_names() -> None:
    class UnencodableString(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot encode")

    class UnencodableNameTool(EchoTool):
        name = UnencodableString("echo")

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool name could not be inspected"):
        registry.register(UnencodableNameTool())


def test_registry_rejects_unstringable_tool_names() -> None:
    class UnstringableString(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

        def __str__(self) -> str:
            raise RuntimeError("name unavailable")

    class UnstringableNameTool(EchoTool):
        name = UnstringableString("echo")

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool name could not be inspected"):
        registry.register(UnstringableNameTool())


def test_registry_rejects_unstrippable_tool_names() -> None:
    class UnstrippableString(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("name unavailable")

    class UnstrippableNameTool(EchoTool):
        name = UnstrippableString("echo")

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool name could not be inspected"):
        registry.register(UnstrippableNameTool())


def test_registry_rejects_non_string_tool_descriptions() -> None:
    class InvalidDescriptionTool(EchoTool):
        description = 123

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool echo description must be a string"):
        registry.register(InvalidDescriptionTool())


@pytest.mark.parametrize(
    ("description", "error"),
    [
        (" ", "Tool echo description cannot be empty"),
        (" Echo text. ", "Tool echo description cannot contain surrounding whitespace"),
    ],
)
def test_registry_rejects_invalid_tool_descriptions(
    description: str,
    error: str,
) -> None:
    class InvalidDescriptionTool(EchoTool):
        pass

    InvalidDescriptionTool.description = description
    registry = ToolRegistry()

    with pytest.raises(ValueError, match=error):
        registry.register(InvalidDescriptionTool())


def test_registry_rejects_oversized_tool_description() -> None:
    class OversizedDescriptionTool(EchoTool):
        description = "x" * 65_537

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool echo description exceeds maximum size"):
        registry.register(OversizedDescriptionTool())


def test_registry_rejects_unencodable_tool_descriptions() -> None:
    class UnencodableString(str):
        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot encode")

    class UnencodableDescriptionTool(EchoTool):
        description = UnencodableString("Echo text.")

    registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="Tool echo description could not be inspected",
    ):
        registry.register(UnencodableDescriptionTool())


def test_registry_rejects_unstringable_tool_descriptions() -> None:
    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("description unavailable")

    class UnstringableDescriptionTool(EchoTool):
        description = UnstringableString("Echo text.")

    registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="Tool echo description could not be inspected",
    ):
        registry.register(UnstringableDescriptionTool())


def test_registry_rejects_unstrippable_tool_descriptions() -> None:
    class UnstrippableString(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("description unavailable")

    class UnstrippableDescriptionTool(EchoTool):
        description = UnstrippableString("Echo text.")

    registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="Tool echo description could not be inspected",
    ):
        registry.register(UnstrippableDescriptionTool())


@pytest.mark.parametrize(
    ("risk", "error"),
    [
        (123, "Tool echo risk must be a string"),
        (" ", "Tool echo risk cannot be empty"),
        (" read ", "Tool echo risk cannot contain whitespace"),
        (
            "network",
            "Tool echo risk must be one of: interactive, read, shell, write",
        ),
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


def test_registry_rejects_uninspectable_tool_risk_metadata() -> None:
    class UnencodableString(str):
        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot encode")

    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("risk unavailable")

    class UnstrippableString(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("risk unavailable")

    cases = [
        UnencodableString("read"),
        UnstringableString("read"),
        UnstrippableString("read"),
    ]
    for risk in cases:
        class InvalidRiskTool(EchoTool):
            pass

        InvalidRiskTool.risk = risk
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="Tool echo risk could not be inspected"):
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
        (
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["missing"],
            },
            "schema required entry must reference a property: missing",
        ),
        (
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "additionalProperties": "false",
            },
            "schema additionalProperties must be a boolean",
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


def test_registry_rejects_recursive_args_schema() -> None:
    schema: dict[str, object] = {"type": "object", "properties": {}}
    properties = schema["properties"]
    assert isinstance(properties, dict)
    properties["self"] = schema

    class RecursiveSchemaTool(EchoTool):
        args_schema = schema

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="schema property self cannot be recursive"):
        registry.register(RecursiveSchemaTool())


def test_registry_rejects_overly_deep_args_schema() -> None:
    schema: dict[str, object] = {"type": "object", "properties": {}}
    current = schema
    for depth in range(80):
        child = {"type": "object", "properties": {}}
        properties = current["properties"]
        assert isinstance(properties, dict)
        properties[f"level_{depth}"] = child
        current = child

    class DeepSchemaTool(EchoTool):
        args_schema = schema

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="schema exceeds maximum depth"):
        registry.register(DeepSchemaTool())


def test_registry_rejects_oversized_args_schema() -> None:
    class WideSchemaTool(EchoTool):
        args_schema = {
            "type": "object",
            "properties": {
                f"field_{index}": {
                    "type": "string",
                    "description": "x" * 300,
                }
                for index in range(9_000)
            },
        }

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="schema exceeds maximum size"):
        registry.register(WideSchemaTool())


def test_registry_rejects_uninspectable_args_schema() -> None:
    class ExplodingProperties(dict):
        def items(self):  # type: ignore[override]
            raise RuntimeError("properties unavailable")

    class UninspectableSchemaTool(EchoTool):
        args_schema = {
            "type": "object",
            "properties": ExplodingProperties(
                {"text": {"type": "string"}},
            ),
        }

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool echo schema could not be inspected"):
        registry.register(UninspectableSchemaTool())


def test_registry_rejects_uninspectable_args_schema_property_names() -> None:
    class UnencodableString(str):
        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot encode")

    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("property name unavailable")

    cases = [UnencodableString("text"), UnstringableString("text")]
    for property_name in cases:
        class InvalidSchemaTool(EchoTool):
            args_schema = {
                "type": "object",
                "properties": {property_name: {"type": "string"}},
            }

        registry = ToolRegistry()

        with pytest.raises(
            ValueError,
            match="schema property names could not be inspected",
        ):
            registry.register(InvalidSchemaTool())


def test_registry_rejects_uninspectable_args_schema_required_entries() -> None:
    class UnencodableString(str):
        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot encode")

    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("required entry unavailable")

    cases = [UnencodableString("text"), UnstringableString("text")]
    for required_entry in cases:
        class InvalidSchemaTool(EchoTool):
            args_schema = {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": [required_entry],
            }

        registry = ToolRegistry()

        with pytest.raises(
            ValueError,
            match="schema required entries could not be inspected",
        ):
            registry.register(InvalidSchemaTool())


def test_registry_rejects_uninspectable_args_schema_type_values() -> None:
    class UnencodableString(str):
        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot encode")

    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("schema type unavailable")

    cases = [
        (
            {"type": UnencodableString("object"), "properties": {}},
            "schema type could not be inspected",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": UnstringableString("string")}},
            },
            "schema property value type could not be inspected",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "value": {"type": ["string", UnencodableString("null")]}
                },
            },
            "schema property value type entries could not be inspected",
        ),
    ]
    for candidate_schema, expected_error in cases:
        class InvalidSchemaTool(EchoTool):
            args_schema = candidate_schema

        registry = ToolRegistry()

        with pytest.raises(ValueError, match=expected_error):
            registry.register(InvalidSchemaTool())


def test_registry_rejects_uninspectable_args_schema_descriptions() -> None:
    class UnstrippableString(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("description unavailable")

    class InvalidSchemaTool(EchoTool):
        args_schema = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": UnstrippableString("Text to echo."),
                }
            },
        }

    registry = ToolRegistry()

    with pytest.raises(
        ValueError,
        match="schema property text description could not be inspected",
    ):
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
                        "type": "strng",
                    }
                },
            },
            "schema property text type must be one of: "
            "array, boolean, integer, null, number, object, string",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": ["string", "strng"],
                    }
                },
            },
            "schema property text type must be one of: "
            "array, boolean, integer, null, number, object, string",
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
                    "text": {
                        "type": "string",
                        "description": 123,
                    }
                },
            },
            "schema property text description must be a string",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": " ",
                    }
                },
            },
            "schema property text description cannot be empty",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "properties": {"child": {"type": "string"}},
                    }
                },
            },
            "schema property text properties require object type",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "required": [],
                    }
                },
            },
            "schema property text required requires object type",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "additionalProperties": False,
                    }
                },
            },
            "schema property text additionalProperties require object type",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": "fast",
                    }
                },
            },
            "schema property mode enum must be a list",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": [],
                    }
                },
            },
            "schema property mode enum cannot be empty",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["fast", 1],
                    }
                },
            },
            "schema property mode enum values must match declared type",
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
                    "settings": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                        "required": ["missing"],
                    }
                },
            },
            "schema property settings required entry must reference "
            "a property: missing",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "settings": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                        "additionalProperties": {"type": "string"},
                    }
                },
            },
            "schema property settings additionalProperties must be a boolean",
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
        (
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "items": {"type": "string"},
                    }
                },
            },
            "schema property text items require array type",
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


def test_registry_normalizes_tool_metadata_text_subclasses() -> None:
    class StickyMetadataTool(EchoTool):
        name = StickyString("echo")
        description = StickyString("Echo text.")
        risk = StickyString("read")

    registry = ToolRegistry([StickyMetadataTool()])

    tool = registry.get("echo")
    schema = registry.to_openai_tools()[0]

    assert tool.name == "echo"
    assert type(tool.name) is str
    assert tool.description == "Echo text."
    assert type(tool.description) is str
    assert tool.risk == "read"
    assert type(tool.risk) is str
    assert schema["name"] == "echo"
    assert type(schema["name"]) is str
    assert schema["description"] == "Echo text."
    assert type(schema["description"]) is str


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


def test_registry_snapshots_tool_schema_on_register() -> None:
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    class MutableSchemaTool(EchoTool):
        args_schema = schema

    tool = MutableSchemaTool()
    registry = ToolRegistry([tool])

    schema["properties"]["text"]["type"] = "integer"
    schema["required"].append("extra")

    exported = registry.to_openai_tools()[0]["parameters"]
    assert exported == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    assert tool.args_schema == exported


def test_registry_keeps_schema_snapshot_when_registered_tool_mutates() -> None:
    original_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    mutated_schema = {
        "type": "object",
        "properties": {"text": {"type": "integer"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    tool = EchoTool()
    registry = ToolRegistry([tool])
    tool.args_schema = mutated_schema

    assert registry.to_openai_tools()[0]["parameters"] == original_schema
    assert registry.get("echo").args_schema == original_schema


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


def test_load_tools_from_module_rejects_too_many_tools(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "too_many_tools.py"
    _write_many_tool_module(module_path, 129)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(
        ValueError,
        match="Cannot load more than 128 tools from modules",
    ):
        load_tools_from_modules(["too_many_tools"])


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


def test_load_tools_from_package_reimports_current_package_path(
    tmp_path,
    monkeypatch,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_pkg = first_dir / "custom_pack"
    second_pkg = second_dir / "custom_pack"
    first_pkg.mkdir(parents=True)
    second_pkg.mkdir(parents=True)
    (first_pkg / "__init__.py").write_text("", encoding="utf-8")
    (second_pkg / "__init__.py").write_text("", encoding="utf-8")
    _write_tool_module(first_pkg / "tools.py", "first_package_tool")
    _write_tool_module(second_pkg / "tools.py", "second_package_tool")

    monkeypatch.syspath_prepend(str(first_dir))
    assert [tool.name for tool in load_tools_from_modules(["custom_pack.tools"])] == [
        "first_package_tool"
    ]

    monkeypatch.syspath_prepend(str(second_dir))
    assert [tool.name for tool in load_tools_from_modules(["custom_pack.tools"])] == [
        "second_package_tool"
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


def test_load_tools_from_module_reports_unstringable_register_tools_failures(
    tmp_path,
    monkeypatch,
) -> None:
    module_path = tmp_path / "unstringable_register_tools.py"
    module_path.write_text(
        "\n".join(
            [
                "class UnstringableError(Exception):",
                "    def __str__(self):",
                "        raise RuntimeError('message unavailable')",
                "",
                "def register_tools(registry):",
                "    raise UnstringableError()",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(
        ValueError,
        match=(
            "Tool module unstringable_register_tools register_tools failed: "
            "UnstringableError"
        ),
    ):
        load_tools_from_modules(["unstringable_register_tools"])


def test_load_tools_from_module_requires_tool_provider(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "empty_tools.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="must define TOOLS or register_tools"):
        load_tools_from_modules(["empty_tools"])


def test_load_tools_from_module_reports_missing_module() -> None:
    with pytest.raises(ValueError, match="Tool module not found: missing_tools"):
        load_tools_from_modules(["missing_tools"])


def test_load_tools_from_module_reports_unstringable_missing_module_name() -> None:
    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("module name unavailable")

    with pytest.raises(ValueError, match="Tool module not found: missing_tools"):
        load_tools_from_modules([UnstringableString("missing_tools")])


def test_load_tools_from_module_rejects_non_string_module_name() -> None:
    with pytest.raises(ValueError, match="Tool module name must be a string"):
        load_tools_from_modules([123])


def test_load_tools_from_modules_rejects_too_many_module_names() -> None:
    with pytest.raises(ValueError, match="Cannot load more than 128 tool modules"):
        load_tools_from_modules([f"module_{index}" for index in range(129)])


def test_load_tools_from_modules_accepts_module_name_iterable_without_length(
    tmp_path,
    monkeypatch,
) -> None:
    class ExplodingModuleNameList(list):
        def __len__(self) -> int:
            raise RuntimeError("module count unavailable")

    module_path = tmp_path / "custom_tools.py"
    _write_tool_module(module_path, "custom_echo")
    monkeypatch.syspath_prepend(str(tmp_path))

    tools = load_tools_from_modules(ExplodingModuleNameList(["custom_tools"]))

    assert [tool.name for tool in tools] == ["custom_echo"]


def test_load_tools_from_module_rejects_oversized_module_name() -> None:
    with pytest.raises(ValueError, match="Tool module name exceeds 512 bytes"):
        load_tools_from_modules(["a" * 513])


def test_load_tools_from_module_rejects_unencodable_module_name() -> None:
    class UnencodableString(str):
        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot encode")

    with pytest.raises(
        ValueError,
        match="Tool module name could not be inspected",
    ):
        load_tools_from_modules([UnencodableString("custom_tools")])


@pytest.mark.parametrize(
    "method_name",
    [
        "strip",
        "split",
        "iter",
    ],
)
def test_load_tools_from_module_rejects_uninspectable_module_name(
    method_name: str,
) -> None:
    class UninspectableString(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if method_name == "strip":
                raise RuntimeError("module name unavailable")
            return super().strip(*args, **kwargs)

        def split(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if method_name == "split":
                raise RuntimeError("module name unavailable")
            return super().split(*args, **kwargs)

        def __iter__(self):  # type: ignore[no-untyped-def]
            if method_name == "iter":
                raise RuntimeError("module name unavailable")
            return super().__iter__()

    with pytest.raises(
        ValueError,
        match="Tool module name could not be inspected",
    ):
        load_tools_from_modules([UninspectableString("custom_tools")])


def test_load_tools_from_module_rejects_blank_module_name() -> None:
    with pytest.raises(ValueError, match="Tool module name cannot be blank"):
        load_tools_from_modules(["   "])


def test_load_tools_from_module_rejects_module_name_whitespace() -> None:
    with pytest.raises(ValueError, match="Tool module name cannot contain whitespace"):
        load_tools_from_modules([" custom_tools"])


def test_load_tools_from_module_rejects_internal_module_name_whitespace() -> None:
    with pytest.raises(ValueError, match="Tool module name cannot contain whitespace"):
        load_tools_from_modules(["custom tools"])


def test_load_tools_from_module_rejects_control_characters_in_module_name() -> None:
    with pytest.raises(
        ValueError,
        match="Tool module name cannot contain control characters",
    ):
        load_tools_from_modules(["custom_tools\x00"])


@pytest.mark.parametrize(
    "module_name",
    [
        "../tools.py",
        ".custom_tools",
        "custom_tools.",
        "custom..tools",
        "custom-tools",
    ],
)
def test_load_tools_from_module_rejects_malformed_module_names(
    module_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Tool module name must be a dotted Python module path",
    ):
        load_tools_from_modules([module_name])


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


def _write_many_tool_module(path, count: int) -> None:
    path.write_text(
        "\n".join(
            [
                "from agentskeleton.tools.base import Tool, ToolResult",
                "",
                "class CustomTool(Tool):",
                "    name = 'bulk_tool'",
                "    description = 'Custom tool.'",
                "    risk = 'read'",
                "    args_schema = {'type': 'object', 'properties': {}}",
                "    def execute(self, args, context):",
                "        return ToolResult(success=True, summary='ok')",
                "",
                f"TOOLS = [CustomTool() for _ in range({count})]",
            ]
        ),
        encoding="utf-8",
    )
