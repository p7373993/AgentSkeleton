import json
from pathlib import Path

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
