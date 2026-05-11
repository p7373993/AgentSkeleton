import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from agentskeleton.tools.base import Tool

TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SUPPORTED_TOOL_RISKS = ("interactive", "read", "shell", "write")
SUPPORTED_JSON_SCHEMA_TYPES = (
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise ValueError("Registered value must be a Tool")
        if not isinstance(tool.name, str):
            raise ValueError("Tool name must be a string")
        name = tool.name.strip()
        if not name:
            raise ValueError("Tool name cannot be empty")
        if name != tool.name:
            raise ValueError("Tool name cannot contain whitespace")
        if TOOL_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError("Tool name must match [A-Za-z0-9_-]+")
        if not isinstance(tool.description, str):
            raise ValueError(f"Tool {name} description must be a string")
        if not tool.description.strip():
            raise ValueError(f"Tool {name} description cannot be empty")
        if tool.description.strip() != tool.description:
            raise ValueError(
                f"Tool {name} description cannot contain surrounding whitespace"
            )
        if not isinstance(tool.risk, str):
            raise ValueError(f"Tool {name} risk must be a string")
        if not tool.risk.strip():
            raise ValueError(f"Tool {name} risk cannot be empty")
        if tool.risk.strip() != tool.risk:
            raise ValueError(f"Tool {name} risk cannot contain whitespace")
        if tool.risk not in SUPPORTED_TOOL_RISKS:
            raise ValueError(
                f"Tool {name} risk must be one of: "
                f"{', '.join(SUPPORTED_TOOL_RISKS)}"
            )
        _validate_args_schema(tool)
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.args_schema),
            }
            for tool in self.all()
        ]


def _validate_args_schema(tool: Tool) -> None:
    schema = tool.args_schema
    if not isinstance(schema, Mapping):
        raise ValueError(f"Tool {tool.name} schema must be a mapping")
    if schema.get("type") != "object":
        raise ValueError(f"Tool {tool.name} schema type must be object")
    _validate_schema_node(tool.name, schema, "schema")


def _validate_schema_node(tool_name: str, schema: Mapping, label: str) -> None:
    description = schema.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise ValueError(f"Tool {tool_name} {label} description must be a string")
        if not description.strip():
            raise ValueError(f"Tool {tool_name} {label} description cannot be empty")

    schema_type = schema.get("type")
    if schema_type is not None:
        if isinstance(schema_type, list):
            if not all(isinstance(item, str) for item in schema_type):
                raise ValueError(
                    f"Tool {tool_name} {label} type entries must be strings"
                )
            if any(item not in SUPPORTED_JSON_SCHEMA_TYPES for item in schema_type):
                raise ValueError(
                    f"Tool {tool_name} {label} type must be one of: "
                    f"{', '.join(SUPPORTED_JSON_SCHEMA_TYPES)}"
                )
        elif not isinstance(schema_type, str):
            raise ValueError(
                f"Tool {tool_name} {label} type must be a string or list of strings"
            )
        elif schema_type not in SUPPORTED_JSON_SCHEMA_TYPES:
            raise ValueError(
                f"Tool {tool_name} {label} type must be one of: "
                f"{', '.join(SUPPORTED_JSON_SCHEMA_TYPES)}"
            )

    enum = schema.get("enum")
    if enum is not None and not isinstance(enum, list):
        raise ValueError(f"Tool {tool_name} {label} enum must be a list")

    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        raise ValueError(f"Tool {tool_name} {label} properties must be a mapping")
    if isinstance(properties, Mapping):
        for property_name, property_schema in properties.items():
            if not isinstance(property_name, str):
                raise ValueError(
                    f"Tool {tool_name} {label} property names must be strings"
                )
            if not isinstance(property_schema, Mapping):
                raise ValueError(
                    f"Tool {tool_name} {label} property {property_name} "
                    "must be a mapping"
                )
            _validate_schema_node(
                tool_name,
                property_schema,
                f"{label} property {property_name}",
            )
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list):
            raise ValueError(f"Tool {tool_name} {label} required must be a list")
        if not all(isinstance(item, str) for item in required):
            raise ValueError(
                f"Tool {tool_name} {label} required entries must be strings"
            )
        property_names = set(properties) if isinstance(properties, Mapping) else set()
        for item in required:
            if item not in property_names:
                raise ValueError(
                    f"Tool {tool_name} {label} required entry must reference "
                    f"a property: {item}"
                )

    additional_properties = schema.get("additionalProperties")
    if additional_properties is not None and not isinstance(
        additional_properties,
        bool,
    ):
        raise ValueError(
            f"Tool {tool_name} {label} additionalProperties must be a boolean"
        )

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError(f"Tool {tool_name} {label} items must be a mapping")
        _validate_schema_node(tool_name, items, f"{label} items")
