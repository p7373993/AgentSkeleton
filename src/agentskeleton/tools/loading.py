import importlib

from agentskeleton.tools.base import Tool
from agentskeleton.tools.registry import ToolRegistry


def load_tools_from_modules(module_names: list[str] | None) -> list[Tool]:
    tools: list[Tool] = []
    for module_name in module_names or []:
        module = importlib.import_module(module_name)

        module_tools = getattr(module, "TOOLS", None)
        register_tools = getattr(module, "register_tools", None)
        if module_tools is None and register_tools is None:
            raise ValueError(
                f"Tool module {module_name} must define TOOLS or register_tools"
            )

        if module_tools is not None:
            tools.extend(_coerce_tool_list(module_name, module_tools))

        if register_tools is not None:
            registry = ToolRegistry()
            register_tools(registry)
            tools.extend(registry.all())

    return tools


def _coerce_tool_list(module_name: str, module_tools: object) -> list[Tool]:
    if not isinstance(module_tools, list):
        raise ValueError(f"Tool module {module_name} TOOLS must be a list")

    tools: list[Tool] = []
    for tool in module_tools:
        if not isinstance(tool, Tool):
            raise ValueError(
                f"Tool module {module_name} TOOLS contains a non-Tool value"
            )
        tools.append(tool)
    return tools
