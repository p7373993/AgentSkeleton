from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4


@dataclass(frozen=True)
class ServerRecord:
    server_id: str
    pid: int
    host: str
    port: int
    root: str
    path: str
    url: str
    log_path: str
    error_log_path: str
    status: str
    started_at: str
    stopped_at: str | None = None
    run_id: str | None = None

    def to_dict(self, running: bool | None = None) -> dict[str, object]:
        payload = asdict(self)
        if running is not None:
            payload["running"] = running
        return payload


class ServerManager:
    def __init__(self, workspace: Path, logs_dir: Path) -> None:
        self.workspace = workspace.resolve()
        self.logs_dir = logs_dir
        self.servers_dir = self.logs_dir / "servers"
        self.logs_path = self.servers_dir / "logs"
        self.registry_path = self.servers_dir / "servers.json"

    def start_static_server(
        self,
        root: str = ".",
        path: str = "",
        port: int = 8000,
        host: str = "127.0.0.1",
        run_id: str | None = None,
        server_id: str | None = None,
    ) -> ServerRecord:
        root_path = self._resolve_root(root)
        check_path = _normalize_url_path(path)
        if check_path:
            check_file = (root_path / check_path).resolve()
            if not _is_relative_to(check_file, root_path) or not check_file.is_file():
                raise ValueError(f"Server check path does not exist: {path}")

        selected_port = self._find_port(host, port)
        server_id = server_id or f"srv-{uuid4().hex[:12]}"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        out_path = self.logs_path / f"{server_id}.out.log"
        err_path = self.logs_path / f"{server_id}.err.log"
        stdout = out_path.open("ab")
        stderr = err_path.open("ab")
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "http.server",
                    str(selected_port),
                    "--bind",
                    host,
                    "-d",
                    str(root_path),
                ],
                cwd=self.workspace,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                creationflags=_creation_flags(),
                start_new_session=os.name != "nt",
            )
        finally:
            stdout.close()
            stderr.close()

        url = f"http://{host}:{selected_port}/{check_path}" if check_path else (
            f"http://{host}:{selected_port}/"
        )
        if not self._wait_for_http(url):
            self._terminate_process(process.pid)
            stderr_text = _read_tail(err_path)
            raise ValueError(
                "Server did not become reachable"
                + (f": {stderr_text}" if stderr_text else "")
            )

        record = ServerRecord(
            server_id=server_id,
            pid=process.pid,
            host=host,
            port=selected_port,
            root=str(root_path),
            path=check_path,
            url=url,
            log_path=str(out_path),
            error_log_path=str(err_path),
            status="running",
            started_at=datetime.now(tz=UTC).isoformat(),
            run_id=run_id,
        )
        self._upsert_record(record)
        return record

    def list_servers(self) -> list[ServerRecord]:
        return self._load_records()

    def stop_server(self, server_id: str) -> ServerRecord:
        records = self._load_records()
        record = self._find_record(records, server_id)
        if self.is_process_running(record.pid):
            self._terminate_process(record.pid)
        stopped = ServerRecord(
            **{
                **record.to_dict(),
                "status": "stopped",
                "stopped_at": datetime.now(tz=UTC).isoformat(),
            }
        )
        updated_records = [
            stopped if item.server_id == record.server_id else item
            for item in records
        ]
        self._save_records(updated_records)
        return stopped

    def restart_server(self, server_id: str) -> ServerRecord:
        record = self.stop_server(server_id)
        return self.start_static_server(
            root=_relative_to_workspace(Path(record.root), self.workspace),
            path=record.path,
            port=record.port,
            host=record.host,
            run_id=record.run_id,
            server_id=record.server_id,
        )

    def is_process_running(self, pid: int) -> bool:
        return _is_process_running(pid)

    def _resolve_root(self, root: str) -> Path:
        root_text = str(root)
        root_path = (self.workspace / root_text).resolve()
        if not _is_relative_to(root_path, self.workspace):
            raise ValueError(f"Server root escapes workspace: {root}")
        if not root_path.exists():
            raise ValueError(f"Server root does not exist: {root}")
        if not root_path.is_dir():
            raise ValueError(f"Server root is not a directory: {root}")
        return root_path

    def _find_port(self, host: str, requested_port: int) -> int:
        if requested_port <= 0 or requested_port > 65535:
            raise ValueError("Server port must be between 1 and 65535")
        for port in range(requested_port, 65536):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((host, port))
                except OSError:
                    continue
                return port
        raise ValueError("No available server port found")

    def _wait_for_http(self, url: str, timeout_seconds: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urlopen(url, timeout=1) as response:
                    return 200 <= response.status < 400
            except (HTTPException, OSError, URLError):
                time.sleep(0.1)
        return False

    def _load_records(self) -> list[ServerRecord]:
        if not self.registry_path.is_file():
            return []
        rows = json.loads(self.registry_path.read_text(encoding="utf-8") or "[]")
        return [ServerRecord(**row) for row in rows if isinstance(row, dict)]

    def _save_records(self, records: list[ServerRecord]) -> None:
        self.servers_dir.mkdir(parents=True, exist_ok=True)
        payload = [record.to_dict() for record in records]
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _upsert_record(self, record: ServerRecord) -> None:
        records = self._load_records()
        self._save_records(
            [item for item in records if item.server_id != record.server_id] + [record]
        )

    def _find_record(
        self,
        records: list[ServerRecord],
        server_id: str,
    ) -> ServerRecord:
        for record in records:
            if record.server_id == server_id:
                return record
        raise ValueError(f"Server not found: {server_id}")

    def _terminate_process(self, pid: int) -> None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def _normalize_url_path(path: str) -> str:
    path_text = str(path).strip().replace("\\", "/").lstrip("/")
    if path_text in {"", "."}:
        return ""
    return path_text


def _relative_to_workspace(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return "."


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(
            handle,
            ctypes.byref(exit_code),
        ):
            return False
        return exit_code.value == 259
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _read_tail(path: Path, max_chars: int = 1000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:].strip()
