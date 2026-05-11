from typing import Any, ClassVar
from uuid import uuid4

from agentskeleton.policy.paths import PathSecurityError, resolve_workspace_path
from agentskeleton.tools.base import Tool, ToolContext, ToolResult


def _error(summary: str, error: str) -> ToolResult:
    return ToolResult(success=False, summary=summary, error=error)


def _path_exists(path) -> bool:
    return path.exists()


def _path_is_dir(path) -> bool:
    return path.is_dir()


def _path_is_file(path) -> bool:
    return path.is_file()


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
        requested = str(args["path"])
        try:
            path = resolve_workspace_path(context.workspace, requested)
        except PathSecurityError as exc:
            return _error(str(exc), "Path escapes workspace")

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
        try:
            entries = [
                {
                    "name": child.name,
                    "type": "directory" if child.is_dir() else "file",
                }
                for child in children
            ]
        except OSError:
            return _error(
                f"Directory listing failed: {requested}",
                "Directory listing failed",
            )
        return ToolResult(
            success=True,
            payload={"path": requested, "entries": entries},
            summary=f"Listed {len(entries)} entries in {requested}",
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
        requested = str(args["path"])
        try:
            path = resolve_workspace_path(context.workspace, requested)
        except PathSecurityError as exc:
            return _error(str(exc), "Path escapes workspace")

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
            raw = path.read_bytes()
        except OSError:
            return _error(f"File read failed: {requested}", "File read failed")
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
        requested = str(args["path"])
        content = str(args["content"])
        create_parent_dirs = bool(args.get("create_parent_dirs", False))
        try:
            path = resolve_workspace_path(context.workspace, requested)
        except PathSecurityError as exc:
            return _error(str(exc), "Path escapes workspace")

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
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
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
