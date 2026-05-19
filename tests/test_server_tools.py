import urllib.request
from pathlib import Path

from agentskeleton.tools.base import ToolContext
from agentskeleton.tools.server import (
    ListServersTool,
    StartStaticServerTool,
    StopServerTool,
)


def test_start_static_server_tool_returns_verified_url_and_persists_record(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<p>tool server</p>", encoding="utf-8")
    context = ToolContext(workspace=tmp_path, logs_dir=tmp_path / "runs")

    result = StartStaticServerTool().execute(
        {"root": "site", "path": "index.html"},
        context,
    )

    try:
        assert result.success is True
        assert result.payload["status_code"] == 200
        assert result.payload["url"].endswith("/index.html")
        assert urllib.request.urlopen(result.payload["url"], timeout=5).status == 200

        listed = ListServersTool().execute({}, context)
        assert listed.success is True
        assert listed.payload["servers"][0]["server_id"] == result.payload["server_id"]
    finally:
        StopServerTool().execute({"server_id": result.payload["server_id"]}, context)


def test_start_static_server_tool_reports_missing_file(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()

    result = StartStaticServerTool().execute(
        {"root": "site", "path": "missing.html"},
        ToolContext(workspace=tmp_path, logs_dir=tmp_path / "runs"),
    )

    assert result.success is False
    assert result.error == "Server start failed"
    assert "Server check path does not exist" in result.summary
