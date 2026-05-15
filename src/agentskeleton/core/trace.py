from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from rich.console import Console

from agentskeleton.logging.run_logger import redact

MAX_TRACE_JSON_CHARS = 1_000


@dataclass(frozen=True)
class TraceEvent:
    name: str
    payload: dict[str, Any]


class TraceSink(Protocol):
    def emit(self, name: str, payload: dict[str, Any]) -> None:
        """Emit an operational trace event."""


class NullTraceSink:
    def emit(self, name: str, payload: dict[str, Any]) -> None:
        pass


class MemoryTraceSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append(TraceEvent(name=name, payload=_snapshot_payload(payload)))


class ConsoleTraceSink:
    def __init__(self, console: Console) -> None:
        self.console = console
        self._last_instructions: str | None = None

    def emit(self, name: str, payload: dict[str, Any]) -> None:
        line = self._format(name, payload)
        if line:
            self.console.print(line, markup=False)

    def _format(self, name: str, payload: dict[str, Any]) -> str:
        if name == "run_started":
            model = preview(payload.get("model"), max_chars=120)
            session = payload.get("session")
            session_part = (
                f" session={preview(session, max_chars=120)}"
                if _has_trace_value(session)
                else ""
            )
            resumed = " resumed" if payload.get("resumed") else ""
            return f"[run] model={model}{session_part}{resumed}"
        if name == "step_started":
            return f"[step] {preview(payload.get('step'), max_chars=80)}"
        if name == "llm_request":
            instructions = payload.get("instructions_preview")
            lines = []
            if _has_trace_value(instructions):
                instructions_text = preview(instructions)
                if instructions_text != self._last_instructions:
                    self._last_instructions = instructions_text
                    lines.append(f"[llm sys] {instructions_text}")
            lines.append(
                "[llm ->] "
                f"input={preview(payload.get('input_preview'))!r} "
                f"tools={_compact_json(payload.get('tool_names', []))}"
            )
            return "\n".join(lines)
        if name == "llm_response":
            calls = payload.get("function_calls")
            if calls is not None and _has_trace_value(calls):
                return f"[llm <-] tool_calls={_compact_json(calls)}"
            return (
                "[llm <-] "
                f"final={preview(payload.get('final_preview'))!r}"
            )
        if name == "model_action":
            return (
                f"[action] {preview(payload.get('tool_name'), max_chars=120)} "
                f"{_compact_json(payload.get('arguments', {}))}"
            )
        if name == "policy_decision":
            return (
                f"[policy] {preview(payload.get('tool_name'), max_chars=120)} "
                f"{preview(payload.get('outcome'), max_chars=80)} - "
                f"{preview(payload.get('reason'))}"
            )
        if name == "tool_started":
            return (
                f"[tool ->] {preview(payload.get('tool_name'), max_chars=120)} "
                f"{_compact_json(payload.get('arguments', {}))}"
            )
        if name == "tool_finished":
            return (
                f"[tool <-] {preview(payload.get('tool_name'), max_chars=120)} "
                f"success={payload.get('success')} "
                f"summary={preview(payload.get('summary'))!r}"
            )
        if name == "run_finished":
            return f"[done] status={preview(payload.get('status'), max_chars=80)}"
        return f"[trace] {name} {_compact_json(payload)}"


def preview(value: Any, max_chars: int = 500) -> str:
    if isinstance(value, str):
        redacted = redact(value)
        text = redacted if isinstance(redacted, str) else str(redacted)
    else:
        text = _compact_json(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _compact_json(value: Any) -> str:
    text = json.dumps(redact(value), ensure_ascii=False, default=str)
    if len(text) <= MAX_TRACE_JSON_CHARS:
        return text
    omitted = len(text) - MAX_TRACE_JSON_CHARS
    return f"{text[:MAX_TRACE_JSON_CHARS]}... [truncated {omitted} characters]"


def _has_trace_value(value: Any) -> bool:
    try:
        return bool(value)
    except Exception:
        return True


def _snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return deepcopy(payload)
    except Exception:
        return {
            _copy_or_original(key): _copy_or_original(value)
            for key, value in payload.items()
        }


def _copy_or_original(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value
