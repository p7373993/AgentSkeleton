from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import unquote

from agentskeleton.tools.base import Tool, ToolContext, ToolResult

MAX_QUERY_LIMIT = 200
DEFAULT_QUERY_LIMIT = 50
MAX_TEXT_PREVIEW_CHARS = 4096
MAX_SEARCH_RESULTS = 50
MAX_JSONL_INSPECT_ROWS = 100
SQL_TIMEOUT_SECONDS = 2.0
WRITE_SQL_RE = re.compile(
    r"\b("
    r"alter|attach|create|delete|detach|drop|insert|pragma|replace|reindex|"
    r"update|vacuum"
    r")\b",
    re.IGNORECASE,
)


def _schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required is not None:
        schema["required"] = required
    return schema


class ListDataSourcesTool(Tool):
    name: ClassVar[str] = "list_data_sources"
    description: ClassVar[str] = "List read-only data sources under data/."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema({})

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        data_root = _data_root(context)
        if not data_root.is_dir():
            return _error("Data directory not found: data", "Data path invalid")
        sources = []
        for path in sorted(
            item for item in data_root.iterdir() if _is_data_source(item)
        ):
            kind = _kind_for_path(path)
            item: dict[str, Any] = {
                "path": _workspace_path(context, path),
                "name": path.name,
                "kind": kind,
                "bytes": path.stat().st_size,
            }
            if kind == "sqlite":
                with _connect_readonly(path) as con:
                    item["tables"] = _table_names(con)
            sources.append(item)
        return ToolResult(
            success=True,
            payload={"sources": sources, "count": len(sources)},
            summary=f"Found {len(sources)} data source(s)",
        )


class InspectSqliteTool(Tool):
    name: ClassVar[str] = "inspect_sqlite"
    description: ClassVar[str] = "Inspect tables and columns in a data/ SQLite file."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {"path": {"type": "string", "description": "SQLite path under data/."}},
        ["path"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_or_error = _sqlite_path_arg(args, context)
        if isinstance(path_or_error, ToolResult):
            return path_or_error
        with _connect_readonly(path_or_error) as con:
            tables = [
                {
                    "name": table,
                    "row_count": _row_count(con, table),
                    "columns": _columns(con, table),
                }
                for table in _table_names(con)
            ]
        return ToolResult(
            success=True,
            payload={"path": _workspace_path(context, path_or_error), "tables": tables},
            summary=f"Inspected {len(tables)} table(s)",
        )


class DescribeSqliteTableTool(Tool):
    name: ClassVar[str] = "describe_sqlite_table"
    description: ClassVar[str] = "Describe one SQLite table under data/."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "path": {"type": "string", "description": "SQLite path under data/."},
            "table": {"type": "string", "description": "Table name."},
        },
        ["path", "table"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        table = str(args.get("table", "")).strip()
        path_or_error = _sqlite_path_arg(args, context)
        if isinstance(path_or_error, ToolResult):
            return path_or_error
        with _connect_readonly(path_or_error) as con:
            if table not in _table_names(con):
                return _error(f"SQLite table not found: {table}", "Table not found")
            payload = {
                "path": _workspace_path(context, path_or_error),
                "table": table,
                "row_count": _row_count(con, table),
                "columns": _columns(con, table),
            }
        return ToolResult(success=True, payload=payload, summary=f"Described {table}")


class SampleSqliteTableTool(Tool):
    name: ClassVar[str] = "sample_sqlite_table"
    description: ClassVar[str] = "Sample rows from one SQLite table under data/."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "path": {"type": "string", "description": "SQLite path under data/."},
            "table": {"type": "string", "description": "Table name."},
            "limit": {"type": "integer", "description": "Maximum rows."},
        },
        ["path", "table"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        table = str(args.get("table", "")).strip()
        limit = _limit(args.get("limit", DEFAULT_QUERY_LIMIT))
        path_or_error = _sqlite_path_arg(args, context)
        if isinstance(path_or_error, ToolResult):
            return path_or_error
        with _connect_readonly(path_or_error) as con:
            if table not in _table_names(con):
                return _error(f"SQLite table not found: {table}", "Table not found")
            rows = _query_rows(con, f'select * from "{table}"', limit)
        return ToolResult(
            success=True,
            payload={"table": table, **rows},
            summary=f"Sampled {len(rows['rows'])} row(s) from {table}",
        )


class QuerySqliteReadOnlyTool(Tool):
    name: ClassVar[str] = "query_sqlite_readonly"
    description: ClassVar[str] = "Run a limited read-only SELECT/WITH query."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "path": {"type": "string", "description": "SQLite path under data/."},
            "query": {"type": "string", "description": "SELECT or WITH SQL."},
            "limit": {"type": "integer", "description": "Maximum rows."},
        },
        ["path", "query"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(args.get("query", "")).strip()
        unsafe = _unsafe_sql_reason(query)
        if unsafe is not None:
            return _error(unsafe, "Unsafe SQL")
        limit = _limit(args.get("limit", DEFAULT_QUERY_LIMIT))
        path_or_error = _sqlite_path_arg(args, context)
        if isinstance(path_or_error, ToolResult):
            return path_or_error
        try:
            with _connect_readonly(path_or_error) as con:
                rows = _query_rows(con, query, limit)
        except sqlite3.Error as exc:
            return _error(f"SQLite query failed: {type(exc).__name__}", "Query failed")
        return ToolResult(
            success=True,
            payload={"path": _workspace_path(context, path_or_error), **rows},
            summary=f"Returned {len(rows['rows'])} row(s)",
        )


class PreviewTextDataTool(Tool):
    name: ClassVar[str] = "preview_text_data"
    description: ClassVar[str] = "Preview a Markdown or text data file."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "path": {"type": "string", "description": "Text path under data/."},
            "max_chars": {"type": "integer", "description": "Maximum characters."},
        },
        ["path"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_or_error = _data_path_arg(args, context, {".md", ".txt"})
        if isinstance(path_or_error, ToolResult):
            return path_or_error
        max_chars = min(_limit(args.get("max_chars", 1000)), MAX_TEXT_PREVIEW_CHARS)
        text = path_or_error.read_text(encoding="utf-8", errors="replace")
        preview = text[:max_chars]
        return ToolResult(
            success=True,
            payload={
                "path": _workspace_path(context, path_or_error),
                "preview": preview,
                "truncated": len(text) > len(preview),
            },
            summary=f"Previewed {len(preview)} character(s)",
        )


class SearchTextDataTool(Tool):
    name: ClassVar[str] = "search_text_data"
    description: ClassVar[str] = "Search Markdown/text data files under data/."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "query": {"type": "string", "description": "Case-insensitive text."},
            "path": {"type": "string", "description": "Optional data text path."},
            "limit": {"type": "integer", "description": "Maximum matches."},
        },
        ["query"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(args.get("query", "")).strip().lower()
        if not query:
            return _error("Search query cannot be blank", "Search invalid")
        limit = min(_limit(args.get("limit", 20)), MAX_SEARCH_RESULTS)
        paths_or_error = _text_search_paths(args, context, {".md", ".txt"})
        if isinstance(paths_or_error, ToolResult):
            return paths_or_error
        matches = _search_text_paths(context, paths_or_error, query, limit)
        return ToolResult(
            success=True,
            payload={"matches": matches, "count": len(matches)},
            summary=f"Found {len(matches)} text match(es)",
        )


class InspectJsonlTool(Tool):
    name: ClassVar[str] = "inspect_jsonl"
    description: ClassVar[str] = "Inspect JSONL keys and sample rows under data/."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {"path": {"type": "string", "description": "JSONL path under data/."}},
        ["path"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_or_error = _data_path_arg(args, context, {".jsonl"})
        if isinstance(path_or_error, ToolResult):
            return path_or_error
        keys: dict[str, str] = {}
        samples = []
        for row_number, row in _iter_jsonl(path_or_error):
            if len(samples) < 3:
                samples.append(row)
            if isinstance(row, dict):
                for key, value in row.items():
                    keys.setdefault(str(key), type(value).__name__)
            if row_number >= MAX_JSONL_INSPECT_ROWS:
                break
        return ToolResult(
            success=True,
            payload={
                "path": _workspace_path(context, path_or_error),
                "keys": dict(sorted(keys.items())),
                "samples": samples,
            },
            summary=f"Inspected {len(keys)} JSONL key(s)",
        )


class SearchJsonlTool(Tool):
    name: ClassVar[str] = "search_jsonl"
    description: ClassVar[str] = "Search JSONL rows under data/."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "path": {"type": "string", "description": "JSONL path under data/."},
            "query": {"type": "string", "description": "Case-insensitive text."},
            "limit": {"type": "integer", "description": "Maximum matches."},
        },
        ["path", "query"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_or_error = _data_path_arg(args, context, {".jsonl"})
        if isinstance(path_or_error, ToolResult):
            return path_or_error
        query = str(args.get("query", "")).strip().lower()
        if not query:
            return _error("Search query cannot be blank", "Search invalid")
        limit = min(_limit(args.get("limit", 20)), MAX_SEARCH_RESULTS)
        matches = []
        for line_number, row in _iter_jsonl(path_or_error):
            row_text = json.dumps(row, ensure_ascii=False).lower()
            if query not in row_text:
                continue
            matches.append({"line": line_number, "row": row})
            if len(matches) >= limit:
                break
        return ToolResult(
            success=True,
            payload={"matches": matches, "count": len(matches)},
            summary=f"Found {len(matches)} JSONL match(es)",
        )


class SearchPartsTool(Tool):
    name: ClassVar[str] = "search_parts"
    description: ClassVar[str] = "Search PDM parts by keyword."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "query": {"type": "string", "description": "Part keyword."},
            "limit": {"type": "integer", "description": "Maximum parts."},
        },
        ["query"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        query = f"%{str(args.get('query', '')).strip()}%"
        limit = _limit(args.get("limit", 20))
        with _domain_db(context, "onepart-metadata-units.sqlite") as con:
            rows = _fetchall(
                con,
                """
                select part_id, title, part_number, file_extension, material,
                       opfamily, mass_kg
                from parts
                where title like ? or part_number like ? or part_id like ?
                order by title
                limit ?
                """,
                (query, query, query, limit),
            )
        return ToolResult(
            success=True,
            payload={"parts": rows, "count": len(rows)},
            summary=f"Found {len(rows)} part(s)",
        )


class GetPartDetailTool(Tool):
    name: ClassVar[str] = "get_part_detail"
    description: ClassVar[str] = "Get PDM part detail with metadata and ERP context."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {"part_id": {"type": "string", "description": "PDM part id."}},
        ["part_id"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        part_id = str(args.get("part_id", "")).strip()
        with _domain_db(context, "onepart-metadata-units.sqlite") as con:
            part = _fetchone(con, "select * from parts where part_id = ?", (part_id,))
            meta = _fetchall(
                con,
                """
                select meta_key, display_name, value_index, meta_value
                from part_meta
                where part_id = ?
                order by meta_key, value_index
                limit 50
                """,
                (part_id,),
            )
        erp = _erp_context(context, part_id)
        return ToolResult(
            success=part is not None,
            payload={"part": part, "metadata": meta, **erp},
            summary="Loaded part detail" if part else f"Part not found: {part_id}",
            error=None if part else "Part not found",
        )


class GetPartRelationsTool(Tool):
    name: ClassVar[str] = "get_part_relations"
    description: ClassVar[str] = "Get BOM/related relations for a PDM part."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "part_id": {"type": "string", "description": "PDM part id."},
            "relation_type": {
                "type": "string",
                "description": "Optional relation type.",
            },
            "limit": {"type": "integer", "description": "Maximum relations."},
        },
        ["part_id"],
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        part_id = str(args.get("part_id", "")).strip()
        relation_type = str(args.get("relation_type", "")).strip()
        limit = _limit(args.get("limit", 50))
        query = "select * from part_relations where src_part_id = ?"
        params: list[Any] = [part_id]
        if relation_type:
            query += " and relation_type = ?"
            params.append(relation_type)
        query += " order by relation_type, dst_title limit ?"
        params.append(limit)
        with _domain_db(context, "onepart-relations.sqlite") as con:
            rows = _fetchall(con, query, tuple(params))
        return ToolResult(
            success=True,
            payload={"relations": rows, "count": len(rows)},
            summary=f"Found {len(rows)} relation(s)",
        )


class GetErpItemByPartTool(Tool):
    name: ClassVar[str] = "get_erp_item_by_part"
    description: ClassVar[str] = "Get ERP item and inventory context for a part."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "part_id": {"type": "string", "description": "PDM part id."},
            "item_id": {"type": "string", "description": "ERP item id."},
        },
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        part_id = str(args.get("part_id", "")).strip()
        item_id = str(args.get("item_id", "")).strip()
        erp = _erp_context(context, part_id=part_id, item_id=item_id)
        return ToolResult(
            success=bool(erp["erp_items"]),
            payload=erp,
            summary=f"Found {len(erp['erp_items'])} ERP item(s)",
            error=None if erp["erp_items"] else "ERP item not found",
        )


class GetProcurementRisksTool(Tool):
    name: ClassVar[str] = "get_procurement_risks"
    description: ClassVar[str] = "List ERP procurement signals and supply risks."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "risk_level": {"type": "string", "description": "Optional risk level."},
            "item_id": {"type": "string", "description": "Optional item id."},
            "limit": {"type": "integer", "description": "Maximum signals."},
        },
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        risk_level = str(args.get("risk_level", "")).strip()
        item_id = str(args.get("item_id", "")).strip()
        limit = _limit(args.get("limit", 20))
        query = "select * from erp_procurement_signal where 1=1"
        params: list[Any] = []
        if risk_level:
            query += " and risk_level = ?"
            params.append(risk_level)
        if item_id:
            query += " and item_id = ?"
            params.append(item_id)
        query += " order by risk_level desc, net_requirement_qty desc limit ?"
        params.append(limit)
        with _domain_db(context, "mock-erp.sqlite") as con:
            rows = _fetchall(con, query, tuple(params))
        return ToolResult(
            success=True,
            payload={"signals": rows, "count": len(rows)},
            summary=f"Found {len(rows)} procurement signal(s)",
        )


class GetAssetServiceHistoryTool(Tool):
    name: ClassVar[str] = "get_asset_service_history"
    description: ClassVar[str] = "Get customer, asset, ticket, and action history."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "asset_id": {"type": "string", "description": "Installed asset id."},
            "customer_id": {"type": "string", "description": "Customer id."},
            "limit": {"type": "integer", "description": "Maximum tickets/actions."},
        },
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        asset_id = str(args.get("asset_id", "")).strip()
        customer_id = str(args.get("customer_id", "")).strip()
        limit = _limit(args.get("limit", 20))
        with _domain_db(context, "mock-as.sqlite") as con:
            customer = None
            if customer_id:
                customer = _fetchone(
                    con,
                    "select * from customer_master where customer_id = ?",
                    (customer_id,),
                )
            asset = None
            if asset_id:
                asset = _fetchone(
                    con,
                    "select * from installed_base where asset_id = ?",
                    (asset_id,),
                )
                if asset and not customer:
                    customer = _fetchone(
                        con,
                        "select * from customer_master where customer_id = ?",
                        (asset["customer_id"],),
                    )
            tickets, actions = _service_history_rows(con, asset_id, customer_id, limit)
        return ToolResult(
            success=True,
            payload={
                "customer": customer,
                "asset": asset,
                "tickets": tickets,
                "actions": actions,
            },
            summary=f"Found {len(tickets)} service ticket(s)",
        )


class GetProjectRisksTool(Tool):
    name: ClassVar[str] = "get_project_risks"
    description: ClassVar[str] = "Get project staffing risks."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = _schema(
        {
            "project_id": {"type": "string", "description": "Optional project id."},
            "limit": {"type": "integer", "description": "Maximum risks."},
        },
    )

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        project_id = str(args.get("project_id", "")).strip()
        limit = _limit(args.get("limit", 20))
        query = "select * from task_staffing_risk"
        params: list[Any] = []
        if project_id:
            query += " where project_id = ?"
            params.append(project_id)
        query += " order by risk_score desc limit ?"
        params.append(limit)
        with _domain_db(context, "mock-project-ops.sqlite") as con:
            rows = _fetchall(con, query, tuple(params))
        return ToolResult(
            success=True,
            payload={"risks": rows, "count": len(rows)},
            summary=f"Found {len(rows)} project risk(s)",
        )


def _error(summary: str, error: str) -> ToolResult:
    return ToolResult(success=False, summary=summary, error=error)


def _data_root(context: ToolContext) -> Path:
    return (context.workspace / "data").resolve()


def _data_path_arg(
    args: dict[str, Any],
    context: ToolContext,
    suffixes: set[str],
) -> Path | ToolResult:
    raw_path = str(args.get("path", "")).strip()
    if not raw_path:
        return _error("Data path cannot be blank", "Data path invalid")
    path = _resolve_data_path(context, raw_path)
    if isinstance(path, ToolResult):
        return path
    if not path.is_file():
        return _error(f"Data file not found: {raw_path}", "Data path invalid")
    if path.suffix.lower() not in suffixes:
        return _error(f"Data file type not supported: {raw_path}", "Data path invalid")
    return path


def _sqlite_path_arg(args: dict[str, Any], context: ToolContext) -> Path | ToolResult:
    return _data_path_arg(args, context, {".sqlite", ".db"})


def _resolve_data_path(context: ToolContext, raw_path: str) -> Path | ToolResult:
    raw_path = unquote(str(raw_path).strip()).replace("\\", "/")
    if Path(raw_path).is_absolute():
        return _error("Data path must be relative to data/", "Data path invalid")
    data_root = _data_root(context)
    candidate = (
        (context.workspace / raw_path).resolve()
        if raw_path == "data" or raw_path.startswith("data/")
        else (data_root / raw_path).resolve()
    )
    if not _is_relative_to(candidate, data_root):
        return _error("Data path escapes data/", "Data path invalid")
    return candidate


def _workspace_path(context: ToolContext, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(context.workspace.resolve())).replace(
            "\\",
            "/",
        )
    except ValueError:
        return str(path)


def _kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".sqlite", ".db"}:
        return "sqlite"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix in {".md", ".txt"}:
        return "text"
    return "other"


def _is_data_source(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower().endswith(("-wal", "-shm", "-journal")):
        return False
    return _kind_for_path(path) != "other"


def _connect_readonly(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    deadline = time.monotonic() + SQL_TIMEOUT_SECONDS

    def progress() -> int:
        return int(time.monotonic() > deadline)

    con.set_progress_handler(progress, 1000)
    return con


def _domain_db(context: ToolContext, file_name: str) -> sqlite3.Connection:
    path = _data_root(context) / file_name
    return _connect_readonly(path)


def _table_names(con: sqlite3.Connection) -> list[str]:
    return [
        row["name"]
        for row in con.execute(
            """
            select name from sqlite_master
            where type = 'table' and name not like 'sqlite_%'
            order by name
            """
        )
    ]


def _columns(con: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [
        {"name": row["name"], "type": row["type"], "notnull": bool(row["notnull"])}
        for row in con.execute(f'pragma table_info("{table}")')
    ]


def _row_count(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f'select count(*) from "{table}"').fetchone()[0])


def _unsafe_sql_reason(query: str) -> str | None:
    if not query:
        return "SQL query cannot be blank"
    normalized = query.strip().rstrip(";").strip()
    if ";" in normalized:
        return "SQL query cannot contain multiple statements"
    first = normalized.split(None, 1)[0].lower() if normalized.split() else ""
    if first not in {"select", "with"}:
        return "Only SELECT/WITH queries are allowed"
    if WRITE_SQL_RE.search(normalized):
        return "SQL query contains a blocked keyword"
    return None


def _query_rows(con: sqlite3.Connection, query: str, limit: int) -> dict[str, Any]:
    normalized = query.strip().rstrip(";")
    limited_query = f"select * from ({normalized}) as limited_query limit ?"
    rows = _fetchall(con, limited_query, (limit + 1,))
    return {
        "rows": rows[:limit],
        "row_count": len(rows[:limit]),
        "limit": limit,
        "truncated": len(rows) > limit,
    }


def _fetchall(
    con: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    return [dict(row) for row in con.execute(query, params).fetchall()]


def _fetchone(
    con: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] = (),
) -> dict[str, Any] | None:
    row = con.execute(query, params).fetchone()
    return dict(row) if row is not None else None


def _limit(value: object) -> int:
    if isinstance(value, bool):
        return DEFAULT_QUERY_LIMIT
    if isinstance(value, int):
        return max(1, min(value, MAX_QUERY_LIMIT))
    if isinstance(value, str) and value.strip().isdigit():
        return max(1, min(int(value), MAX_QUERY_LIMIT))
    return DEFAULT_QUERY_LIMIT


def _text_search_paths(
    args: dict[str, Any],
    context: ToolContext,
    suffixes: set[str],
) -> list[Path] | ToolResult:
    raw_path = str(args.get("path", "")).strip()
    if raw_path:
        path = _data_path_arg(args, context, suffixes)
        return path if isinstance(path, ToolResult) else [path]
    data_root = _data_root(context)
    return [
        path
        for path in sorted(data_root.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def _search_text_paths(
    context: ToolContext,
    paths: list[Path],
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    matches = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            1,
        ):
            if query not in line.lower():
                continue
            matches.append(
                {
                    "path": _workspace_path(context, path),
                    "line": line_number,
                    "text": line[:500],
                }
            )
            if len(matches) >= limit:
                return matches
    return matches


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                continue


def _erp_context(
    context: ToolContext,
    part_id: str = "",
    item_id: str = "",
) -> dict[str, list[dict[str, Any]]]:
    query = "select * from erp_item_master where 1=1"
    params: list[Any] = []
    if part_id:
        query += " and part_id = ?"
        params.append(part_id)
    if item_id:
        query += " and item_id = ?"
        params.append(item_id)
    query += " limit 20"
    with _domain_db(context, "mock-erp.sqlite") as con:
        items = _fetchall(con, query, tuple(params))
        item_ids = [item["item_id"] for item in items]
        inventory = _rows_for_ids(con, "erp_inventory_balance", "item_id", item_ids)
        signals = _rows_for_ids(con, "erp_procurement_signal", "item_id", item_ids)
    return {"erp_items": items, "inventory": inventory, "procurement_signals": signals}


def _rows_for_ids(
    con: sqlite3.Connection,
    table: str,
    column: str,
    ids: list[str],
) -> list[dict[str, Any]]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    return _fetchall(
        con,
        f'select * from "{table}" where "{column}" in ({placeholders}) limit 50',
        tuple(ids),
    )


def _service_history_rows(
    con: sqlite3.Connection,
    asset_id: str,
    customer_id: str,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query = "select * from as_ticket where 1=1"
    params: list[Any] = []
    if asset_id:
        query += " and asset_id = ?"
        params.append(asset_id)
    if customer_id:
        query += " and customer_id = ?"
        params.append(customer_id)
    query += " order by ticket_date desc limit ?"
    params.append(limit)
    tickets = _fetchall(con, query, tuple(params))
    ticket_ids = [ticket["ticket_id"] for ticket in tickets]
    actions = _rows_for_ids(con, "as_action_history", "ticket_id", ticket_ids)
    return tickets, actions


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
