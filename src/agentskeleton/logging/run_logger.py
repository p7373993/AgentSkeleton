from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SECRET_PATTERNS = [
    re.compile(r"(OPENAI_API_KEY\s*=\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]+"),
]


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in SECRET_PATTERNS:
            if pattern.groups:
                redacted = pattern.sub(r"\1[REDACTED]", redacted)
            else:
                redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    return value


class RunLogger:
    def __init__(self, logs_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        today = datetime.now(tz=UTC).strftime("%Y%m%d")
        self.path = logs_dir / today / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, step: int, payload: dict[str, Any]) -> None:
        event = {
            "type": event_type,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "payload": redact(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
