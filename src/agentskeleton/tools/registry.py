import json
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
MAX_TOOL_SCHEMA_DEPTH = 64
MAX_TOOL_SCHEMA_BYTES = 2_097_152
MAX_TOOL_DESCRIPTION_BYTES = 65_536
MAX_TOOL_NAME_BYTES = 512
MAX_REGISTERED_TOOLS = 128


def _utf8_size(value: str) -> int | None:
    try:
        return len(value.encode("utf-8"))
    except Exception:
        return None


def _safe_text(value: object) -> str:
    try:
        text = str(value)
    except Exception:
        return "<uninspectable>"
    return str.__str__(text)


def _is_uninspectable_text(value: str) -> bool:
    return _utf8_size(value) is None or _safe_text(value) == "<uninspectable>"


def _safe_strip(value: str) -> str | None:
    try:
        stripped = value.strip()
    except Exception:
        return None
    if not isinstance(stripped, str):
        return None
    return str.__str__(stripped)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        for tool in () if tools is None else tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise ValueError("Registered value must be a Tool")
        if not isinstance(tool.name, str):
            raise ValueError("Tool name must be a string")
        name = _safe_strip(tool.name)
        if name is None:
            raise ValueError("Tool name could not be inspected")
        if not name:
            raise ValueError("Tool name cannot be empty")
        if name != tool.name:
            raise ValueError("Tool name cannot contain whitespace")
        if TOOL_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError("Tool name must match [A-Za-z0-9_-]+")
        name_bytes = _utf8_size(name)
        if _is_uninspectable_text(tool.name):
            raise ValueError("Tool name could not be inspected")
        if name_bytes > MAX_TOOL_NAME_BYTES:
            raise ValueError("Tool name exceeds maximum size")
        if not isinstance(tool.description, str):
            raise ValueError(f"Tool {name} description must be a string")
        description = _safe_strip(tool.description)
        if description is None:
            raise ValueError(f"Tool {name} description could not be inspected")
        if not description:
            raise ValueError(f"Tool {name} description cannot be empty")
        if description != tool.description:
            raise ValueError(
                f"Tool {name} description cannot contain surrounding whitespace"
            )
        description_bytes = _utf8_size(tool.description)
        if _is_uninspectable_text(tool.description):
            raise ValueError(f"Tool {name} description could not be inspected")
        if description_bytes > MAX_TOOL_DESCRIPTION_BYTES:
            raise ValueError(f"Tool {name} description exceeds maximum size")
        if not isinstance(tool.risk, str):
            raise ValueError(f"Tool {name} risk must be a string")
        risk = _safe_strip(tool.risk)
        if risk is None:
            raise ValueError(f"Tool {name} risk could not be inspected")
        if not risk:
            raise ValueError(f"Tool {name} risk cannot be empty")
        if risk != tool.risk:
            raise ValueError(f"Tool {name} risk cannot contain whitespace")
        if _is_uninspectable_text(tool.risk):
            raise ValueError(f"Tool {name} risk could not be inspected")
        if risk not in SUPPORTED_TOOL_RISKS:
            raise ValueError(
                f"Tool {name} risk must be one of: "
                f"{', '.join(SUPPORTED_TOOL_RISKS)}"
            )
        schema_snapshot = _validate_args_schema(tool)
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        if len(self._tools) >= MAX_REGISTERED_TOOLS:
            raise ValueError(f"Too many tools registered (max {MAX_REGISTERED_TOOLS})")
        self._schemas[name] = schema_snapshot
        tool.name = name
        tool.description = description
        tool.risk = risk
        tool.args_schema = deepcopy(schema_snapshot)
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tool_with_schema_snapshot(name)
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {_safe_text(name)}") from exc

    def all(self) -> list[Tool]:
        return [self._tool_with_schema_snapshot(name) for name in self._tools]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(self._schemas[tool.name]),
            }
            for tool in self.all()
        ]

    def _tool_with_schema_snapshot(self, name: str) -> Tool:
        tool = self._tools[name]
        tool.args_schema = deepcopy(self._schemas[name])
        return tool


def _validate_args_schema(tool: Tool) -> dict[str, Any]:
    try:
        return _validate_args_schema_inner(tool)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Tool {tool.name} schema could not be inspected") from exc


def _validate_args_schema_inner(tool: Tool) -> dict[str, Any]:
    schema = tool.args_schema
    if not isinstance(schema, Mapping):
        raise ValueError(f"Tool {tool.name} schema must be a mapping")
    if schema.get("type") != "object":
        raise ValueError(f"Tool {tool.name} schema type must be object")
    _validate_schema_size(tool.name, schema)
    _validate_schema_node(tool.name, schema, "schema", set(), 0)
    return deepcopy(dict(schema))


def _validate_schema_size(tool_name: str, schema: Mapping) -> None:
    try:
        size = len(json.dumps(schema, default=str).encode("utf-8"))
    except (TypeError, ValueError, RecursionError):
        return
    if size > MAX_TOOL_SCHEMA_BYTES:
        raise ValueError(f"Tool {tool_name} schema exceeds maximum size")


def _validate_schema_node(
    tool_name: str,
    schema: Mapping,
    label: str,
    seen: set[int],
    depth: int,
) -> None:
    if depth > MAX_TOOL_SCHEMA_DEPTH:
        raise ValueError(f"Tool {tool_name} schema exceeds maximum depth")
    marker = id(schema)
    if marker in seen:
        raise ValueError(f"Tool {tool_name} {label} cannot be recursive")
    seen.add(marker)
    try:
        _validate_schema_node_content(tool_name, schema, label, seen, depth)
    finally:
        seen.remove(marker)


def _validate_schema_node_content(
    tool_name: str,
    schema: Mapping,
    label: str,
    seen: set[int],
    depth: int,
) -> None:
    description = schema.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise ValueError(f"Tool {tool_name} {label} description must be a string")
        stripped_description = _safe_strip(description)
        if stripped_description is None:
            raise ValueError(
                f"Tool {tool_name} {label} description could not be inspected"
            )
        if not stripped_description:
            raise ValueError(f"Tool {tool_name} {label} description cannot be empty")

    schema_type = schema.get("type")
    if schema_type is not None:
        if isinstance(schema_type, list):
            if not all(isinstance(item, str) for item in schema_type):
                raise ValueError(
                    f"Tool {tool_name} {label} type entries must be strings"
                )
            if any(
                _is_uninspectable_text(item) for item in schema_type
            ):
                raise ValueError(
                    f"Tool {tool_name} {label} type entries could not be inspected"
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
        elif _is_uninspectable_text(schema_type):
            raise ValueError(f"Tool {tool_name} {label} type could not be inspected")
        elif schema_type not in SUPPORTED_JSON_SCHEMA_TYPES:
            raise ValueError(
                f"Tool {tool_name} {label} type must be one of: "
                f"{', '.join(SUPPORTED_JSON_SCHEMA_TYPES)}"
            )

    enum = schema.get("enum")
    if enum is not None and not isinstance(enum, list):
        raise ValueError(f"Tool {tool_name} {label} enum must be a list")
    if isinstance(enum, list) and not _has_items(enum):
        raise ValueError(f"Tool {tool_name} {label} enum cannot be empty")
    if isinstance(enum, list) and not all(
        _matches_schema_type(item, schema_type) for item in enum
    ):
        raise ValueError(
            f"Tool {tool_name} {label} enum values must match declared type"
        )

    properties = schema.get("properties")
    if properties is not None:
        if not _schema_type_includes(schema_type, "object"):
            raise ValueError(
                f"Tool {tool_name} {label} properties require object type"
            )
        if not isinstance(properties, Mapping):
            raise ValueError(f"Tool {tool_name} {label} properties must be a mapping")
    if isinstance(properties, Mapping):
        for property_name, property_schema in properties.items():
            if not isinstance(property_name, str):
                raise ValueError(
                    f"Tool {tool_name} {label} property names must be strings"
                )
            if _is_uninspectable_text(property_name):
                raise ValueError(
                    f"Tool {tool_name} {label} property names could not be inspected"
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
                seen,
                depth + 1,
            )
    required = schema.get("required")
    if required is not None:
        if not _schema_type_includes(schema_type, "object"):
            raise ValueError(f"Tool {tool_name} {label} required requires object type")
        if not isinstance(required, list):
            raise ValueError(f"Tool {tool_name} {label} required must be a list")
        if not all(isinstance(item, str) for item in required):
            raise ValueError(
                f"Tool {tool_name} {label} required entries must be strings"
            )
        if any(_is_uninspectable_text(item) for item in required):
            raise ValueError(
                f"Tool {tool_name} {label} required entries could not be inspected"
            )
        property_names = set(properties) if isinstance(properties, Mapping) else set()
        for item in required:
            if item not in property_names:
                raise ValueError(
                    f"Tool {tool_name} {label} required entry must reference "
                    f"a property: {item}"
                )

    additional_properties = schema.get("additionalProperties")
    if additional_properties is not None:
        if not _schema_type_includes(schema_type, "object"):
            raise ValueError(
                f"Tool {tool_name} {label} additionalProperties require object type"
            )
        if not isinstance(additional_properties, bool):
            raise ValueError(
                f"Tool {tool_name} {label} additionalProperties must be a boolean"
            )

    items = schema.get("items")
    if items is not None:
        if not _schema_type_includes(schema_type, "array"):
            raise ValueError(f"Tool {tool_name} {label} items require array type")
        if not isinstance(items, Mapping):
            raise ValueError(f"Tool {tool_name} {label} items must be a mapping")
        _validate_schema_node(tool_name, items, f"{label} items", seen, depth + 1)


def _schema_type_includes(raw_type: object, expected_type: str) -> bool:
    if isinstance(raw_type, str):
        return raw_type == expected_type
    if isinstance(raw_type, list):
        return expected_type in raw_type
    return False


def _has_items(items: Iterable[object]) -> bool:
    for _item in items:
        return True
    return False


def _matches_schema_type(value: object, raw_type: object) -> bool:
    if isinstance(raw_type, str):
        return _matches_single_schema_type(value, raw_type)
    if isinstance(raw_type, list):
        return any(_matches_single_schema_type(value, item) for item in raw_type)
    return True


def _matches_single_schema_type(value: object, expected_type: object) -> bool:
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "null":
        return value is None
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "string":
        return isinstance(value, str)
    return True
