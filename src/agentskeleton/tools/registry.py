import re
from collections.abc import Iterable, Mapping
from typing import Any

from agentskeleton.tools.base import Tool

TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool) -> None:
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
        if not isinstance(tool.risk, str):
            raise ValueError(f"Tool {name} risk must be a string")
        if not tool.risk.strip():
            raise ValueError(f"Tool {name} risk cannot be empty")
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
                "parameters": tool.args_schema,
            }
            for tool in self.all()
        ]


def _validate_args_schema(tool: Tool) -> None:
    schema = tool.args_schema
    if not isinstance(schema, Mapping):
        raise ValueError(f"Tool {tool.name} schema must be a mapping")
    if schema.get("type") != "object":
        raise ValueError(f"Tool {tool.name} schema type must be object")
    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        raise ValueError(f"Tool {tool.name} schema properties must be a mapping")
    if isinstance(properties, Mapping):
        for property_name, property_schema in properties.items():
            if not isinstance(property_name, str):
                raise ValueError(
                    f"Tool {tool.name} schema property names must be strings"
                )
            if not isinstance(property_schema, Mapping):
                raise ValueError(
                    f"Tool {tool.name} schema property {property_name} "
                    "must be a mapping"
                )
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list):
            raise ValueError(f"Tool {tool.name} schema required must be a list")
        if not all(isinstance(item, str) for item in required):
            raise ValueError(
                f"Tool {tool.name} schema required entries must be strings"
            )
