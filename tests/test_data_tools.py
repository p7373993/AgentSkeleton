import json
import sqlite3
from pathlib import Path

from agentskeleton.tools.base import ToolContext
from agentskeleton.tools.data import (
    DescribeSqliteTableTool,
    GetAssetServiceHistoryTool,
    GetErpItemByPartTool,
    GetPartDetailTool,
    GetPartRelationsTool,
    GetProcurementRisksTool,
    GetProjectRisksTool,
    InspectJsonlTool,
    InspectSqliteTool,
    ListDataSourcesTool,
    PreviewTextDataTool,
    QuerySqliteReadOnlyTool,
    SampleSqliteTableTool,
    SearchJsonlTool,
    SearchPartsTool,
    SearchTextDataTool,
)


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _make_data_root(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "notes.md").write_text(
        "# Manufacturing Data\n\nPump bearing and motor notes.\n",
        encoding="utf-8",
    )
    (data / "trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tool": "lookup_part_detail", "partId": "P-001"}),
                json.dumps({"tool": "search", "query": "bearing"}),
            ]
        ),
        encoding="utf-8",
    )

    con = _connect(data / "onepart-metadata-units.sqlite")
    con.executescript(
        """
        create table parts (
          part_id text primary key,
          title text,
          part_number text,
          file_extension text,
          material text,
          opfamily text,
          mass_kg real
        );
        create table part_meta (
          part_id text,
          meta_key text,
          display_name text,
          value_index integer,
          meta_value text,
          collected_at text
        );
        insert into parts values
          ('P-001','Bearing Housing.sldprt','BH-001','sldprt','Steel','solidworks',1.5),
          ('P-002','Motor Assembly.sldasm','MA-002','sldasm','', 'solidworks',7.2);
        insert into part_meta values
          ('P-001','hole_diameters','hole_diameters',0,'10mm','2026-01-01'),
          ('P-001','main_color','main_color',0,'black','2026-01-01');
        """
    )
    con.commit()
    con.close()
    (data / "onepart-metadata-units.sqlite-wal").write_text("", encoding="utf-8")
    (data / "scratch.tmp").write_text("ignore me", encoding="utf-8")

    con = _connect(data / "onepart-relations.sqlite")
    con.executescript(
        """
        create table part_relations (
          src_part_id text,
          relation_type text,
          dst_key text,
          dst_title text,
          dst_url text,
          dst_source text,
          raw_value text,
          collected_at text
        );
        insert into part_relations values (
          'P-001',
          'parent',
          'parent|MA-002',
          'Motor Assembly.sldasm',
          '/parts/MA-002',
          'mock',
          'raw',
          '2026-01-01'
        );
        """
    )
    con.commit()
    con.close()

    con = _connect(data / "mock-erp.sqlite")
    con.executescript(
        """
        create table erp_item_master (
          item_id text primary key,
          part_id text,
          erp_item_code text,
          part_number text,
          item_name text,
          item_category text,
          make_buy_type text,
          lifecycle_status text
        );
        create table erp_inventory_balance (
          inventory_id text,
          item_id text,
          warehouse_code text,
          on_hand_qty integer,
          available_qty integer,
          allocated_qty integer,
          on_order_qty integer,
          safety_stock_qty integer,
          reorder_point_qty integer,
          reorder_qty integer,
          monthly_demand_qty integer,
          inventory_status text
        );
        create table erp_procurement_signal (
          signal_id text,
          item_id text,
          period_month text,
          projected_available_qty integer,
          gross_demand_qty integer,
          net_requirement_qty integer,
          recommended_order_qty integer,
          recommended_order_date text,
          risk_level text,
          driver text
        );
        insert into erp_item_master values (
          'ITEM-001',
          'P-001',
          'ERP-001',
          'BH-001',
          'Bearing Housing',
          'component',
          'BUY',
          'released'
        );
        insert into erp_inventory_balance values
          ('INV-001','ITEM-001','WH-A',5,3,2,10,4,6,20,8,'critical');
        insert into erp_procurement_signal values
          ('SIG-001','ITEM-001','2026-04-01',-2,12,8,20,'2026-03-20','high','bom_demand');
        """
    )
    con.commit()
    con.close()

    con = _connect(data / "mock-as.sqlite")
    con.executescript(
        """
        create table customer_master (
          customer_id text primary key,
          customer_name text,
          industry text,
          region text,
          account_manager_employee_id text
        );
        create table installed_base (
          asset_id text primary key,
          customer_id text,
          project_id text,
          project_code text,
          product_model text,
          serial_number text,
          install_date text,
          warranty_status text,
          site_location text,
          primary_part_id text,
          erp_item_code text
        );
        create table as_ticket (
          ticket_id text,
          customer_id text,
          asset_id text,
          project_id text,
          ticket_date text,
          inquiry_channel text,
          issue_category text,
          issue_subcategory text,
          symptom_text text,
          severity text,
          status text,
          assigned_employee_id text
        );
        create table as_action_history (
          action_id text,
          ticket_id text,
          action_datetime text,
          action_type text,
          action_note text,
          used_part_id text,
          used_item_id text,
          onsite_flag integer,
          resolved_flag integer
        );
        insert into customer_master values (
          'CUST-001',
          'Acme Factory',
          'automotive',
          'KR',
          'EMP-001'
        );
        insert into installed_base values (
          'AST-001',
          'CUST-001',
          'PRJ-001',
          'SSP-001',
          'BPS-8200',
          'SN-1',
          '2026-01-01',
          'active',
          'Seoul',
          'P-001',
          'ERP-001'
        );
        insert into as_ticket values (
          'TKT-001',
          'CUST-001',
          'AST-001',
          'PRJ-001',
          '2026-04-01',
          'phone',
          'bearing_noise',
          'startup',
          'Bearing noise observed',
          'high',
          'closed',
          'EMP-002'
        );
        insert into as_action_history values (
          'ACT-001',
          'TKT-001',
          '2026-04-02',
          'repair',
          'Replaced bearing housing',
          'P-001',
          'ITEM-001',
          1,
          1
        );
        """
    )
    con.commit()
    con.close()

    con = _connect(data / "mock-project-ops.sqlite")
    con.executescript(
        """
        create table task_staffing_risk (
          risk_id text,
          project_id text,
          task_id text,
          employee_id text,
          risk_type text,
          risk_score real,
          risk_reason text,
          detected_at text
        );
        insert into task_staffing_risk values (
          'RISK-001',
          'PRJ-001',
          'TSK-001',
          'EMP-002',
          'business_trip_overlap',
          91.5,
          'Owner unavailable',
          '2026-04-01'
        );
        """
    )
    con.commit()
    con.close()
    return data


def _context(tmp_path: Path) -> ToolContext:
    _make_data_root(tmp_path)
    return ToolContext(workspace=tmp_path)


def test_list_data_sources_inventories_sqlite_markdown_and_jsonl(
    tmp_path: Path,
) -> None:
    result = ListDataSourcesTool().execute({}, _context(tmp_path))

    assert result.success is True
    names = {item["path"] for item in result.payload["sources"]}
    assert "data/onepart-metadata-units.sqlite" in names
    assert "data/notes.md" in names
    assert "data/trace.jsonl" in names
    assert "data/onepart-metadata-units.sqlite-wal" not in names
    assert "data/scratch.tmp" not in names
    sqlite_source = next(
        item
        for item in result.payload["sources"]
        if item["path"] == "data/onepart-metadata-units.sqlite"
    )
    assert sqlite_source["kind"] == "sqlite"
    assert sqlite_source["tables"] == ["part_meta", "parts"]


def test_sqlite_tools_inspect_sample_and_query_without_writes(tmp_path: Path) -> None:
    context = _context(tmp_path)

    inspected = InspectSqliteTool().execute(
        {"path": "data/onepart-metadata-units.sqlite"},
        context,
    )
    described = DescribeSqliteTableTool().execute(
        {"path": "data/onepart-metadata-units.sqlite", "table": "parts"},
        context,
    )
    sampled = SampleSqliteTableTool().execute(
        {"path": "data/onepart-metadata-units.sqlite", "table": "parts", "limit": 1},
        context,
    )
    queried = QuerySqliteReadOnlyTool().execute(
        {
            "path": "data/onepart-metadata-units.sqlite",
            "query": "select part_id, title from parts order by part_id",
            "limit": 1,
        },
        context,
    )

    assert inspected.payload["tables"][0]["name"] == "part_meta"
    assert described.payload["table"] == "parts"
    assert sampled.payload["rows"][0]["part_id"] == "P-001"
    assert queried.payload["rows"] == [
        {"part_id": "P-001", "title": "Bearing Housing.sldprt"}
    ]
    assert queried.payload["truncated"] is True


def test_query_sqlite_rejects_writes_and_paths_outside_data(tmp_path: Path) -> None:
    context = _context(tmp_path)
    outside = tmp_path / "outside.sqlite"
    sqlite3.connect(outside).close()

    write_result = QuerySqliteReadOnlyTool().execute(
        {"path": "data/onepart-metadata-units.sqlite", "query": "delete from parts"},
        context,
    )
    outside_result = InspectSqliteTool().execute(
        {"path": "outside.sqlite"},
        context,
    )

    assert write_result.success is False
    assert write_result.error == "Unsafe SQL"
    assert outside_result.success is False
    assert outside_result.error == "Data path invalid"


def test_text_and_jsonl_tools_preview_search_and_inspect(tmp_path: Path) -> None:
    context = _context(tmp_path)

    preview = PreviewTextDataTool().execute(
        {"path": "data/notes.md", "max_chars": 20},
        context,
    )
    search = SearchTextDataTool().execute(
        {"query": "bearing", "path": "data/notes.md"},
        context,
    )
    inspected = InspectJsonlTool().execute({"path": "data/trace.jsonl"}, context)
    jsonl_search = SearchJsonlTool().execute(
        {"path": "data/trace.jsonl", "query": "lookup_part_detail"},
        context,
    )

    assert preview.payload["preview"] == "# Manufacturing Data"
    assert search.payload["matches"][0]["line"] == 3
    assert inspected.payload["keys"] == {
        "partId": "str",
        "query": "str",
        "tool": "str",
    }
    assert jsonl_search.payload["matches"][0]["row"]["partId"] == "P-001"


def test_manufacturing_shortcuts_return_cross_domain_context(tmp_path: Path) -> None:
    context = _context(tmp_path)

    parts = SearchPartsTool().execute({"query": "bearing"}, context)
    detail = GetPartDetailTool().execute({"part_id": "P-001"}, context)
    relations = GetPartRelationsTool().execute({"part_id": "P-001"}, context)
    erp = GetErpItemByPartTool().execute({"part_id": "P-001"}, context)
    procurement = GetProcurementRisksTool().execute({"risk_level": "high"}, context)
    service = GetAssetServiceHistoryTool().execute({"asset_id": "AST-001"}, context)
    project_risks = GetProjectRisksTool().execute({"project_id": "PRJ-001"}, context)

    assert parts.payload["parts"][0]["part_id"] == "P-001"
    assert detail.payload["part"]["part_number"] == "BH-001"
    assert detail.payload["erp_items"][0]["item_id"] == "ITEM-001"
    assert relations.payload["relations"][0]["dst_title"] == "Motor Assembly.sldasm"
    assert erp.payload["inventory"][0]["inventory_status"] == "critical"
    assert procurement.payload["signals"][0]["risk_level"] == "high"
    assert service.payload["tickets"][0]["issue_category"] == "bearing_noise"
    assert service.payload["actions"][0]["used_part_id"] == "P-001"
    assert project_risks.payload["risks"][0]["risk_score"] == 91.5
