import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentskeleton.logging.run_logger import RunLogger


def test_run_logger_writes_jsonl_event(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")

    logger.log("run_started", step=0, payload={"goal": "test"})

    lines = logger.path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    assert event["run_id"] == "run-1"
    assert event["step"] == 0
    assert event["type"] == "run_started"
    assert event["payload"] == {"goal": "test"}
    assert logger.path.parent.parent == tmp_path


def test_run_logger_sanitizes_run_id_path_segments(tmp_path: Path) -> None:
    today = datetime.now(tz=UTC).strftime("%Y%m%d")
    logger = RunLogger(logs_dir=tmp_path, run_id="../bad id")

    logger.log("run_started", step=0, payload={"goal": "test"})

    expected_path = tmp_path / today / "bad_id.jsonl"
    event = json.loads(expected_path.read_text(encoding="utf-8").splitlines()[0])
    assert logger.path == expected_path
    assert event["run_id"] == "../bad id"
    assert not (tmp_path / "bad id.jsonl").exists()


def test_run_logger_bounds_very_long_run_id_filenames(tmp_path: Path) -> None:
    today = datetime.now(tz=UTC).strftime("%Y%m%d")
    run_ids = ["r" * 319 + "x", "r" * 319 + "y"]

    for index, run_id in enumerate(run_ids):
        RunLogger(logs_dir=tmp_path, run_id=run_id).log(
            "run_started",
            step=0,
            payload={"index": index},
        )

    log_paths = list((tmp_path / today).iterdir())
    events = [
        json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        for path in sorted(log_paths)
    ]
    assert len(log_paths) == 2
    assert all(len(path.stem) <= 120 for path in log_paths)
    assert {event["run_id"] for event in events} == set(run_ids)


@pytest.mark.parametrize(
    ("run_id", "expected_name"),
    [
        ("NUL", "NUL_.jsonl"),
        ("CON.log", "CON.log_.jsonl"),
        ("COM1", "COM1_.jsonl"),
    ],
)
def test_run_logger_suffixes_windows_reserved_run_id_stems(
    tmp_path: Path,
    run_id: str,
    expected_name: str,
) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id=run_id)

    logger.log("run_started", step=0, payload={"goal": "test"})

    assert logger.path.name == expected_name
    assert logger.path.is_file()


def test_run_logger_redacts_obvious_secrets(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")

    logger.log(
        "tool_finished",
        step=1,
        payload={
            "stdout": "OPENAI_API_KEY=sk-testsecret\nAuthorization: Bearer token123"
        },
    )

    raw = logger.path.read_text(encoding="utf-8")
    assert "sk-testsecret" not in raw
    assert "token123" not in raw
    assert "[REDACTED]" in raw


def test_run_logger_redacts_common_key_value_secrets(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")

    logger.log(
        "tool_finished",
        step=1,
        payload={
            "stdout": (
                "api_key=abc123\n"
                "password: hunter2\n"
                "x-api-key: vendor-secret"
            )
        },
    )

    raw = logger.path.read_text(encoding="utf-8")
    assert "abc123" not in raw
    assert "hunter2" not in raw
    assert "vendor-secret" not in raw
    assert raw.count("[REDACTED]") == 3


def test_run_logger_redacts_secret_mapping_values(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")

    logger.log(
        "tool_finished",
        step=1,
        payload={
            "api_key": "abc123",
            "nested": {
                "password": "hunter2",
                "headers": {"Authorization": "Bearer token123"},
            },
        },
    )

    raw = logger.path.read_text(encoding="utf-8")
    assert "abc123" not in raw
    assert "hunter2" not in raw
    assert "token123" not in raw
    assert raw.count("[REDACTED]") == 3


def test_run_logger_redacts_common_token_and_secret_names(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")

    logger.log(
        "tool_finished",
        step=1,
        payload={
            "access_token": "access-token-value",
            "nested": {
                "refresh-token": "refresh-token-value",
                "client_secret": "oauth-client-value",
            },
            "stdout": (
                "GITHUB_TOKEN=ghp-token-value\n"
                "AWS_SECRET_ACCESS_KEY=aws-secret-value"
            ),
            "normal": "visible-value",
        },
    )

    raw = logger.path.read_text(encoding="utf-8")
    assert "access-token-value" not in raw
    assert "refresh-token-value" not in raw
    assert "oauth-client-value" not in raw
    assert "ghp-token-value" not in raw
    assert "aws-secret-value" not in raw
    assert "visible-value" in raw


def test_run_logger_bounds_large_payload_strings(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")
    large_output = "x" * 20_000

    logger.log(
        "tool_finished",
        step=1,
        payload={"stdout": large_output},
    )

    raw = logger.path.read_text(encoding="utf-8")
    event = json.loads(raw.splitlines()[0])
    assert len(event["payload"]["stdout"]) < 5_000
    assert event["payload"]["stdout"].startswith("xxxxxxxxxxxxxxxx")
    assert "[truncated" in event["payload"]["stdout"]
    assert large_output not in raw


def test_run_logger_bounds_wide_payload_values(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")
    payload = {
        "items": list(range(250)),
        **{f"key_{index}": index for index in range(250)},
    }

    logger.log("tool_finished", step=1, payload=payload)

    raw = logger.path.read_text(encoding="utf-8")
    event = json.loads(raw.splitlines()[0])
    stored_payload = event["payload"]
    items = stored_payload["items"]

    assert stored_payload["key_0"] == 0
    assert stored_payload["key_197"] == 197
    assert "key_198" not in stored_payload
    assert "key_249" not in stored_payload
    assert stored_payload["__truncated_items__"] == {
        "truncated": True,
        "items": 251,
        "omitted": 52,
    }
    assert items[:3] == [0, 1, 2]
    assert items[198] == 198
    assert items[199] == {
        "truncated": True,
        "items": 250,
        "omitted": 51,
    }
    assert "key_249" not in raw


def test_run_logger_serializes_non_json_payload_values(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")
    marker = object()

    logger.log(
        "tool_finished",
        step=1,
        payload={
            "workspace": tmp_path,
            "raw": marker,
            42: "numeric key",
        },
    )

    event = json.loads(logger.path.read_text(encoding="utf-8").splitlines()[0])
    assert event["payload"]["workspace"] == str(tmp_path)
    assert event["payload"]["raw"] == str(marker)
    assert event["payload"]["42"] == "numeric key"


def test_run_logger_serializes_recursive_payload_values(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")
    payload = {}
    payload["self"] = payload

    logger.log("tool_finished", step=1, payload={"payload": payload})

    event = json.loads(logger.path.read_text(encoding="utf-8").splitlines()[0])
    assert event["payload"]["payload"] == {"self": "<recursive>"}


def test_run_logger_bounds_deep_payload_values(tmp_path: Path) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")
    payload: dict[str, object] = {}
    current = payload
    for _ in range(1_200):
        child: dict[str, object] = {}
        current["child"] = child
        current = child

    logger.log("tool_finished", step=1, payload={"payload": payload})

    raw = logger.path.read_text(encoding="utf-8")
    assert "<max-depth-exceeded>" in raw


def test_run_logger_reports_date_directory_file(tmp_path: Path) -> None:
    today = datetime.now(tz=UTC).strftime("%Y%m%d")
    date_path = tmp_path / today
    date_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="Run log directory is not a directory"):
        RunLogger(logs_dir=tmp_path, run_id="run-1")


def test_run_logger_reports_directory_exists_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    today = datetime.now(tz=UTC).strftime("%Y%m%d")
    date_path = tmp_path / today
    original_exists = Path.exists

    def fail_date_exists(path: Path) -> bool:
        if path == date_path:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_date_exists)

    with pytest.raises(ValueError) as exc_info:
        RunLogger(logs_dir=tmp_path, run_id="run-1")

    assert str(exc_info.value) == f"Run log directory could not be checked: {date_path}"


def test_run_logger_reports_directory_type_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    today = datetime.now(tz=UTC).strftime("%Y%m%d")
    date_path = tmp_path / today
    date_path.mkdir()
    original_is_dir = Path.is_dir

    def fail_date_is_dir(path: Path) -> bool:
        if path == date_path:
            raise OSError("permission denied")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fail_date_is_dir)

    with pytest.raises(ValueError) as exc_info:
        RunLogger(logs_dir=tmp_path, run_id="run-1")

    assert str(exc_info.value) == f"Run log directory could not be checked: {date_path}"


def test_run_logger_reports_log_path_directory(tmp_path: Path) -> None:
    today = datetime.now(tz=UTC).strftime("%Y%m%d")
    log_path = tmp_path / today / "run-1.jsonl"
    log_path.mkdir(parents=True)
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")

    with pytest.raises(ValueError, match="Run log path is not a file"):
        logger.log("run_started", step=0, payload={"goal": "test"})


def test_run_logger_reports_log_path_exists_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")
    original_exists = Path.exists

    def fail_log_exists(path: Path) -> bool:
        if path == logger.path:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_log_exists)

    with pytest.raises(ValueError) as exc_info:
        logger.log("run_started", step=0, payload={"goal": "test"})

    assert str(exc_info.value) == f"Run log path could not be checked: {logger.path}"


def test_run_logger_reports_log_path_type_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logger = RunLogger(logs_dir=tmp_path, run_id="run-1")
    logger.path.write_text("", encoding="utf-8")
    original_is_file = Path.is_file

    def fail_log_is_file(path: Path) -> bool:
        if path == logger.path:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_log_is_file)

    with pytest.raises(ValueError) as exc_info:
        logger.log("run_started", step=0, payload={"goal": "test"})

    assert str(exc_info.value) == f"Run log path could not be checked: {logger.path}"
