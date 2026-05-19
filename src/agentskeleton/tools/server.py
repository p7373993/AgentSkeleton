from typing import Any, ClassVar

from agentskeleton.server_manager import ServerManager
from agentskeleton.tools.base import Tool, ToolContext, ToolResult


class StartStaticServerTool(Tool):
    name: ClassVar[str] = "start_static_server"
    description: ClassVar[str] = (
        "Start a managed local static HTTP server for workspace files."
    )
    risk: ClassVar[str] = "server"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "root": {
                "type": "string",
                "description": "Workspace-relative directory to serve.",
            },
            "path": {
                "type": "string",
                "description": "Workspace-relative path under root to verify.",
            },
            "port": {
                "type": "integer",
                "description": "Preferred localhost port.",
            },
            "server_id": {
                "type": "string",
                "description": "Optional stable managed server id.",
            },
        },
        "required": ["root", "path"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            record = _manager(context).start_static_server(
                root=str(args.get("root", ".")),
                path=str(args.get("path", "")),
                port=_port(args.get("port", 8000)),
                run_id=context.run_id,
                server_id=_optional_text(args.get("server_id")),
            )
        except ValueError as exc:
            return ToolResult(
                success=False,
                summary=str(exc),
                error="Server start failed",
            )
        payload = record.to_dict(running=True)
        payload["status_code"] = 200
        return ToolResult(
            success=True,
            payload=payload,
            summary=f"Started static server at {record.url}",
        )


class ListServersTool(Tool):
    name: ClassVar[str] = "list_servers"
    description: ClassVar[str] = "List managed local servers."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        manager = _manager(context)
        servers = [
            record.to_dict(running=manager.is_process_running(record.pid))
            for record in manager.list_servers()
        ]
        return ToolResult(
            success=True,
            payload={"servers": servers},
            summary=f"Found {len(servers)} managed server(s)",
        )


class StopServerTool(Tool):
    name: ClassVar[str] = "stop_server"
    description: ClassVar[str] = "Stop a managed local server."
    risk: ClassVar[str] = "server"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "server_id": {
                "type": "string",
                "description": "Managed server id.",
            }
        },
        "required": ["server_id"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            record = _manager(context).stop_server(str(args.get("server_id", "")))
        except ValueError as exc:
            return ToolResult(
                success=False,
                summary=str(exc),
                error="Server stop failed",
            )
        return ToolResult(
            success=True,
            payload=record.to_dict(running=False),
            summary=f"Stopped server {record.server_id}",
        )


class RestartServerTool(Tool):
    name: ClassVar[str] = "restart_server"
    description: ClassVar[str] = "Restart a managed local server."
    risk: ClassVar[str] = "server"
    args_schema: ClassVar[dict[str, Any]] = StopServerTool.args_schema

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            record = _manager(context).restart_server(str(args.get("server_id", "")))
        except ValueError as exc:
            return ToolResult(
                success=False,
                summary=str(exc),
                error="Server restart failed",
            )
        return ToolResult(
            success=True,
            payload=record.to_dict(running=True),
            summary=f"Restarted server {record.server_id} at {record.url}",
        )


def _manager(context: ToolContext) -> ServerManager:
    return ServerManager(workspace=context.workspace, logs_dir=context.logs_dir)


def _port(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Server port must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return 8000


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
