from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MAX_RUN_LOG_STEM_LENGTH = 120
SECRET_PATTERNS = [
    re.compile(r"(OPENAI_API_KEY\s*=\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(
        r"(\b[A-Z0-9_.-]*(?:TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE[_-]?KEY|"
        r"CREDENTIAL|ACCESS[_-]?KEY)[A-Z0-9_.-]*\s*[:=]\s*)[^\s]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:x-)?api[-_]?key\s*[:=]\s*)[^\s]+",
        re.IGNORECASE,
    ),
    re.compile(r"(password\s*[:=]\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]+"),
]
SECRET_KEY_TERMS = {
    "access_key",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
    "x_api_key",
}


def _safe_log_stem(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    safe_value = safe_value or "run"
    if len(safe_value) > _MAX_RUN_LOG_STEM_LENGTH:
        digest = hashlib.sha256(safe_value.encode("utf-8")).hexdigest()[:12]
        prefix_length = _MAX_RUN_LOG_STEM_LENGTH - len(digest) - 1
        safe_value = f"{safe_value[:prefix_length]}-{digest}"
    return safe_value


def redact(value: Any, seen: set[int] | None = None) -> Any:
    seen = seen or set()
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return {
                str(key): "[REDACTED]" if _is_secret_key(key) else redact(item, seen)
                for key, item in value.items()
            }
        finally:
            seen.remove(marker)
    if isinstance(value, list | tuple):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return [redact(item, seen) for item in value]
        finally:
            seen.remove(marker)
    if isinstance(value, str):
        redacted = value
        for pattern in SECRET_PATTERNS:
            if pattern.groups:
                redacted = pattern.sub(r"\1[REDACTED]", redacted)
            else:
                redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    if value is None or isinstance(value, int | float | bool):
        return value
    return str(value)


def _is_secret_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(term in normalized for term in SECRET_KEY_TERMS)


class RunLogger:
    def __init__(self, logs_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        today = datetime.now(tz=UTC).strftime("%Y%m%d")
        self.path = logs_dir / today / f"{_safe_log_stem(run_id)}.jsonl"
        self._ensure_log_directory()

    def log(self, event_type: str, step: int, payload: dict[str, Any]) -> None:
        event = {
            "type": event_type,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "payload": redact(payload),
        }
        try:
            log_exists = self.path.exists()
        except OSError as exc:
            raise ValueError(f"Run log path could not be checked: {self.path}") from exc
        if log_exists:
            try:
                log_is_file = self.path.is_file()
            except OSError as exc:
                raise ValueError(
                    f"Run log path could not be checked: {self.path}"
                ) from exc
            if not log_is_file:
                raise ValueError(f"Run log path is not a file: {self.path}")
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise ValueError(f"Run log could not be written: {self.path}") from exc

    def _ensure_log_directory(self) -> None:
        for candidate in (self.path.parent, *self.path.parent.parents):
            try:
                candidate_exists = candidate.exists()
            except OSError as exc:
                raise ValueError(
                    f"Run log directory could not be checked: {candidate}"
                ) from exc
            if not candidate_exists:
                continue
            try:
                candidate_is_directory = candidate.is_dir()
            except OSError as exc:
                raise ValueError(
                    f"Run log directory could not be checked: {candidate}"
                ) from exc
            if not candidate_is_directory:
                raise ValueError(
                    f"Run log directory is not a directory: {candidate}"
                )
            break
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"Run log directory could not be created: {self.path.parent}"
            ) from exc
