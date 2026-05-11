import importlib
import sys

from agentskeleton.tools.base import Tool
from agentskeleton.tools.registry import ToolRegistry

_PROTECTED_PARENT_MODULES = ("agentskeleton",)
MAX_TOOL_MODULES = 128
MAX_TOOL_MODULE_NAME_BYTES = 512
MAX_LOADED_TOOLS = 128


def load_tools_from_modules(module_names: list[str] | None) -> list[Tool]:
    tools: list[Tool] = []
    names = module_names or []
    if len(names) > MAX_TOOL_MODULES:
        raise ValueError(f"Cannot load more than {MAX_TOOL_MODULES} tool modules")
    for module_name in names:
        _validate_module_name(module_name)
        try:
            _evict_tool_module_cache(module_name)
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or module_name
            if module_name == missing_name or module_name.startswith(
                f"{missing_name}."
            ):
                raise ValueError(f"Tool module not found: {module_name}") from exc
            raise ValueError(
                f"Tool module {module_name} could not import dependency: "
                f"{missing_name}"
            ) from exc
        except ImportError as exc:
            raise ValueError(
                f"Tool module {module_name} could not be imported: {exc}"
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"Tool module {module_name} could not be imported: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        module_tools = getattr(module, "TOOLS", None)
        register_tools = getattr(module, "register_tools", None)
        if module_tools is None and register_tools is None:
            raise ValueError(
                f"Tool module {module_name} must define TOOLS or register_tools"
            )

        if module_tools is not None:
            _append_loaded_tools(tools, _coerce_tool_list(module_name, module_tools))

        if register_tools is not None:
            if not callable(register_tools):
                raise ValueError(
                    f"Tool module {module_name} register_tools must be callable"
                )
            registry = ToolRegistry()
            try:
                register_tools(registry)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    f"Tool module {module_name} register_tools failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            _append_loaded_tools(tools, registry.all())

    return tools


def _validate_module_name(module_name: object) -> None:
    if not isinstance(module_name, str):
        raise ValueError("Tool module name must be a string")
    if len(module_name.encode("utf-8")) > MAX_TOOL_MODULE_NAME_BYTES:
        raise ValueError(f"Tool module name exceeds {MAX_TOOL_MODULE_NAME_BYTES} bytes")
    if not module_name.strip():
        raise ValueError("Tool module name cannot be blank")
    if any(ord(character) < 32 for character in module_name):
        raise ValueError("Tool module name cannot contain control characters")
    if any(character.isspace() for character in module_name):
        raise ValueError("Tool module name cannot contain whitespace")
    if any(not part.isidentifier() for part in module_name.split(".")):
        raise ValueError("Tool module name must be a dotted Python module path")


def _evict_tool_module_cache(module_name: str) -> None:
    importlib.invalidate_caches()
    prefixes = _module_cache_prefixes(module_name)
    for loaded_name in list(sys.modules):
        if any(
            loaded_name == prefix or loaded_name.startswith(f"{prefix}.")
            for prefix in prefixes
        ):
            sys.modules.pop(loaded_name, None)


def _module_cache_prefixes(module_name: str) -> tuple[str, ...]:
    parts = module_name.split(".")
    prefixes = [module_name]
    for index in range(len(parts) - 1, 0, -1):
        parent_name = ".".join(parts[:index])
        if _is_protected_parent_module(parent_name):
            continue
        prefixes.append(parent_name)
    return tuple(prefixes)


def _is_protected_parent_module(module_name: str) -> bool:
    return any(
        module_name == protected or module_name.startswith(f"{protected}.")
        for protected in _PROTECTED_PARENT_MODULES
    )


def _append_loaded_tools(tools: list[Tool], new_tools: list[Tool]) -> None:
    if len(tools) + len(new_tools) > MAX_LOADED_TOOLS:
        raise ValueError(f"Cannot load more than {MAX_LOADED_TOOLS} tools from modules")
    tools.extend(new_tools)


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
