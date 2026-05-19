import json
import urllib.request
from pathlib import Path

from typer.testing import CliRunner

from agentskeleton.cli import app
from agentskeleton.core.actions import FinalAction, ToolCallAction
from agentskeleton.server_manager import ServerManager


def test_servers_command_lists_managed_servers_as_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("cli", encoding="utf-8")
    record = ServerManager(tmp_path, tmp_path / "runs").start_static_server(
        root="site",
        path="index.html",
    )
    runner = CliRunner()

    try:
        result = runner.invoke(app, ["servers", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["servers"][0]["server_id"] == record.server_id
        assert payload["servers"][0]["url"] == record.url
        assert payload["servers"][0]["running"] is True
    finally:
        ServerManager(tmp_path, tmp_path / "runs").stop_server(record.server_id)


def test_stop_server_command_stops_managed_server(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("cli", encoding="utf-8")
    manager = ServerManager(tmp_path, tmp_path / "runs")
    record = manager.start_static_server(root="site", path="index.html")
    runner = CliRunner()

    result = runner.invoke(app, ["stop-server", record.server_id])

    assert result.exit_code == 0
    assert f"Stopped server {record.server_id}" in result.stdout
    assert manager.is_process_running(record.pid) is False


def test_restart_server_command_restarts_managed_server(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("cli", encoding="utf-8")
    manager = ServerManager(tmp_path, tmp_path / "runs")
    record = manager.start_static_server(root="site", path="index.html")
    runner = CliRunner()

    result = runner.invoke(app, ["restart-server", record.server_id, "--json"])

    try:
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["server"]["server_id"] == record.server_id
        assert payload["server"]["pid"] != record.pid
        assert urllib.request.urlopen(payload["server"]["url"], timeout=5).status == 200
    finally:
        ServerManager(tmp_path, tmp_path / "runs").stop_server(record.server_id)


def test_agent_run_can_create_artifact_and_start_verified_server(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "\n".join(
            [
                "permission_profile: trusted",
                "confirm_risky_actions: false",
            ]
        ),
        encoding="utf-8",
    )

    class FakeLLM:
        def __init__(self) -> None:
            self.actions = [
                ToolCallAction(
                    tool_name="write_file",
                    arguments={
                        "path": "tetris/index.html",
                        "content": "<h1>managed tetris</h1>",
                        "create_parent_dirs": True,
                    },
                    call_id="write-html",
                ),
                ToolCallAction(
                    tool_name="start_static_server",
                    arguments={
                        "root": ".",
                        "path": "tetris/index.html",
                        "port": 8000,
                    },
                    call_id="serve-html",
                ),
                FinalAction(text="server ready"),
            ]

        def next_action(self, state, registry):  # noqa: ANN001, ANN201
            return self.actions.pop(0)

    monkeypatch.setattr(
        "agentskeleton.cli.LLMClient",
        lambda config, **kwargs: FakeLLM(),
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["run", "make tetris", "--config", str(config_path), "--quiet"],
    )

    manager = ServerManager(tmp_path, tmp_path / "runs")
    servers = manager.list_servers()
    try:
        assert result.exit_code == 0
        assert "server ready" in result.stdout
        assert (tmp_path / "tetris" / "index.html").is_file()
        assert len(servers) == 1
        assert urllib.request.urlopen(servers[0].url, timeout=5).status == 200
        assert manager.is_process_running(servers[0].pid) is True
    finally:
        for server in servers:
            manager.stop_server(server.server_id)
