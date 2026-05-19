import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agentskeleton.server_manager import ServerManager


def _read_url(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read().decode("utf-8")


def _reserve_port() -> socket.socket:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock


def test_start_static_server_persists_and_serves_file(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>hello server</h1>", encoding="utf-8")
    manager = ServerManager(workspace=tmp_path, logs_dir=tmp_path / "runs")

    record = manager.start_static_server(root="site", path="index.html")

    try:
        status, body = _read_url(record.url)
        assert status == 200
        assert "hello server" in body
        assert manager.is_process_running(record.pid) is True
        assert record.root == str(site.resolve())
        assert record.log_path.endswith(".out.log")
        assert record.error_log_path.endswith(".err.log")
        assert manager.list_servers()[0].server_id == record.server_id
    finally:
        manager.stop_server(record.server_id)

    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(record.url, timeout=1)


def test_start_static_server_chooses_next_port_when_requested_port_is_busy(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("ok", encoding="utf-8")
    manager = ServerManager(workspace=tmp_path, logs_dir=tmp_path / "runs")
    sock = _reserve_port()
    busy_port = sock.getsockname()[1]

    try:
        record = manager.start_static_server(
            root="site",
            path="index.html",
            port=busy_port,
        )
    finally:
        sock.close()

    try:
        assert record.port != busy_port
        assert _read_url(record.url)[0] == 200
    finally:
        manager.stop_server(record.server_id)


def test_start_static_server_reports_missing_root(tmp_path: Path) -> None:
    manager = ServerManager(workspace=tmp_path, logs_dir=tmp_path / "runs")

    with pytest.raises(ValueError, match="Server root does not exist"):
        manager.start_static_server(root="missing", path="index.html")
