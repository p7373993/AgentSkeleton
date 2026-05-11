from typing import Any, ClassVar
from uuid import uuid4

from agentskeleton.policy.paths import PathSecurityError, resolve_workspace_path
from agentskeleton.tools.base import Tool, ToolContext, ToolResult

MAX_READ_FILE_BYTES = 1_048_576
MAX_WRITE_FILE_BYTES = 1_048_576
MAX_LIST_DIR_ENTRIES = 200


def _error(summary: str, error: str) -> ToolResult:
    return ToolResult(success=False, summary=summary, error=error)


def _path_security_error(exc: PathSecurityError) -> ToolResult:
    summary = str(exc)
    error = (
        "Path escapes workspace"
        if summary.startswith("Path escapes workspace")
        else "Path invalid"
    )
    return _error(summary, error)


def _string_arg(
    args: dict[str, Any],
    name: str,
    label: str,
    *,
    allow_blank: bool = False,
    allow_control_chars: bool = False,
) -> str | ToolResult:
    value = args.get(name)
    if not isinstance(value, str):
        return _error(
            f"{label} invalid: {name} must be a string",
            f"{label} invalid",
        )
    if not allow_blank and not value.strip():
        return _error(
            f"{label} invalid: {name} cannot be blank",
            f"{label} invalid",
        )
    if not allow_control_chars and any(ord(character) < 32 for character in value):
        return _error(
            f"{label} invalid: {name} cannot contain control characters",
            f"{label} invalid",
        )
    return value


def _bool_arg(
    args: dict[str, Any],
    name: str,
    default: bool,
    label: str,
) -> bool | ToolResult:
    value = args.get(name, default)
    if not isinstance(value, bool):
        return _error(
            f"{label} invalid: {name} must be a boolean",
            f"{label} invalid",
        )
    return value


def _path_exists(path) -> bool:
    return path.exists()


def _path_is_dir(path) -> bool:
    return path.is_dir()


def _path_is_file(path) -> bool:
    return path.is_file()


def _path_size(path) -> int:
    return path.stat().st_size


class ListDirTool(Tool):
    name: ClassVar[str] = "list_dir"
    description: ClassVar[str] = (
        "List direct children of a workspace-relative directory."
    )
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path."}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        requested = _string_arg(args, "path", "Path")
        if isinstance(requested, ToolResult):
            return requested
        try:
            path = resolve_workspace_path(context.workspace, requested)
        except PathSecurityError as exc:
            return _path_security_error(exc)

        try:
            path_exists = _path_exists(path)
        except OSError:
            return _error(
                f"Directory listing failed: {requested}",
                "Directory listing failed",
            )
        if not path_exists:
            return _error(f"Directory not found: {requested}", "Directory not found")
        try:
            path_is_directory = _path_is_dir(path)
        except OSError:
            return _error(
                f"Directory listing failed: {requested}",
                "Directory listing failed",
            )
        if not path_is_directory:
            return _error(f"Not a directory: {requested}", "Not a directory")

        try:
            children = sorted(path.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            return _error(
                f"Directory listing failed: {requested}",
                "Directory listing failed",
            )
        visible_children = children[:MAX_LIST_DIR_ENTRIES]
        total_entries = len(children)
        try:
            entries = [
                {
                    "name": child.name,
                    "type": "directory" if child.is_dir() else "file",
                }
                for child in visible_children
            ]
        except OSError:
            return _error(
                f"Directory listing failed: {requested}",
                "Directory listing failed",
            )
        listed_entries = len(entries)
        truncated = total_entries > listed_entries
        summary = (
            f"Listed {listed_entries} of {total_entries} entries in {requested}"
            if truncated
            else f"Listed {listed_entries} entries in {requested}"
        )
        return ToolResult(
            success=True,
            payload={
                "path": requested,
                "entries": entries,
                "total_entries": total_entries,
                "truncated": truncated,
            },
            summary=summary,
        )


class ReadFileTool(Tool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "Read a UTF-8 text file inside the workspace."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path."}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        requested = _string_arg(args, "path", "Path")
        if isinstance(requested, ToolResult):
            return requested
        try:
            path = resolve_workspace_path(context.workspace, requested)
        except PathSecurityError as exc:
            return _path_security_error(exc)

        try:
            path_exists = _path_exists(path)
        except OSError:
            return _error(f"File read failed: {requested}", "File read failed")
        if not path_exists:
            return _error(f"File not found: {requested}", "File not found")
        try:
            path_is_file = _path_is_file(path)
        except OSError:
            return _error(f"File read failed: {requested}", "File read failed")
        if not path_is_file:
            return _error(f"Not a file: {requested}", "Not a file")
        try:
            file_size = _path_size(path)
        except OSError:
            return _error(f"File read failed: {requested}", "File read failed")
        if file_size > MAX_READ_FILE_BYTES:
            return _error(f"File too large: {requested}", "File too large")

        try:
            raw = path.read_bytes()
        except OSError:
            return _error(f"File read failed: {requested}", "File read failed")
        if len(raw) > MAX_READ_FILE_BYTES:
            return _error(f"File too large: {requested}", "File too large")
        if b"\x00" in raw:
            return _error(f"Binary file rejected: {requested}", "Binary file rejected")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return _error(f"Binary file rejected: {requested}", "Binary file rejected")

        return ToolResult(
            success=True,
            payload={"path": requested, "content": content},
            summary=f"Read {len(content)} characters from {requested}",
        )


class WriteFileTool(Tool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = "Write UTF-8 text inside the workspace."
    risk: ClassVar[str] = "write"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path."},
            "content": {"type": "string", "description": "Text content to write."},
            "create_parent_dirs": {
                "type": "boolean",
                "description": "Create missing parent directories.",
                "default": False,
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        requested = _string_arg(args, "path", "Path")
        if isinstance(requested, ToolResult):
            return requested
        content = _string_arg(
            args,
            "content",
            "Content",
            allow_blank=True,
            allow_control_chars=True,
        )
        if isinstance(content, ToolResult):
            return content
        create_parent_dirs = _bool_arg(
            args,
            "create_parent_dirs",
            False,
            "Create parent dirs",
        )
        if isinstance(create_parent_dirs, ToolResult):
            return create_parent_dirs
        try:
            path = resolve_workspace_path(context.workspace, requested)
        except PathSecurityError as exc:
            return _path_security_error(exc)

        try:
            parent_exists = _path_exists(path.parent)
        except OSError:
            return _error(f"File write failed: {requested}", "File write failed")
        if not parent_exists:
            if not create_parent_dirs:
                return _error(
                    f"Parent directory not found: {requested}",
                    "Parent directory not found",
                )
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                return _error(f"File write failed: {requested}", "File write failed")
        else:
            try:
                parent_is_directory = _path_is_dir(path.parent)
            except OSError:
                return _error(f"File write failed: {requested}", "File write failed")
            if not parent_is_directory:
                return _error(
                    f"Parent is not a directory: {requested}",
                    "Parent is not a directory",
                )

        try:
            target_exists = _path_exists(path)
        except OSError:
            return _error(f"File write failed: {requested}", "File write failed")
        if target_exists:
            try:
                target_is_file = _path_is_file(path)
            except OSError:
                return _error(f"File write failed: {requested}", "File write failed")
            if not target_is_file:
                return _error(f"Not a file: {requested}", "Not a file")

        encoded = content.encode("utf-8")
        if len(encoded) > MAX_WRITE_FILE_BYTES:
            return _error(f"Content too large: {requested}", "Content too large")

        temp_path = path.parent / f".write-{uuid4().hex}.tmp"
        try:
            temp_path.write_bytes(encoded)
            temp_path.replace(path)
        except OSError:
            return _error(f"File write failed: {requested}", "File write failed")
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

        return ToolResult(
            success=True,
            payload={"path": requested, "bytes_written": len(encoded)},
            summary=f"Wrote {len(encoded)} bytes to {requested}",
        )
