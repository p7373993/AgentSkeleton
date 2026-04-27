from pathlib import Path
from typing import Any, ClassVar

from agentskeleton.policy.paths import PathSecurityError, resolve_workspace_path
from agentskeleton.tools.base import Tool, ToolContext, ToolResult


def _error(summary: str, error: str) -> ToolResult:
    return ToolResult(success=False, summary=summary, error=error)


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

        if not path.exists():
            return _error(f"Directory not found: {requested}", "Directory not found")
        if not path.is_dir():
            return _error(f"Not a directory: {requested}", "Not a directory")

        entries = [
            {
                "name": child.name,
                "type": "directory" if child.is_dir() else "file",
            }
            for child in sorted(path.iterdir(), key=lambda item: item.name.lower())
        ]
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

        if not path.exists():
            return _error(f"File not found: {requested}", "File not found")
        if not path.is_file():
            return _error(f"Not a file: {requested}", "Not a file")

        raw = path.read_bytes()
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

        if not path.parent.exists():
            if not create_parent_dirs:
                return _error(
                    f"Parent directory not found: {requested}",
                    "Parent directory not found",
                )
            path.parent.mkdir(parents=True, exist_ok=True)

        encoded = content.encode("utf-8")
        temp_path = Path(f"{path}.tmp")
        temp_path.write_bytes(encoded)
        temp_path.replace(path)

        return ToolResult(
            success=True,
            payload={"path": requested, "bytes_written": len(encoded)},
            summary=f"Wrote {len(encoded)} bytes to {requested}",
        )
