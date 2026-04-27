from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from rich.console import Console


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
        self.events.append(TraceEvent(name=name, payload=payload))


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
            session = payload.get("session")
            session_part = f" session={session}" if session else ""
            resumed = " resumed" if payload.get("resumed") else ""
            return (
                f"[run] model={payload.get('model')}"
                f"{session_part}{resumed}"
            )
        if name == "step_started":
            return f"[step] {payload.get('step')}"
        if name == "llm_request":
            instructions = payload.get("instructions_preview")
            lines = []
            if instructions and instructions != self._last_instructions:
                self._last_instructions = str(instructions)
                lines.append(f"[llm sys] {instructions}")
            lines.append(
                "[llm ->] "
                f"input={payload.get('input_preview')!r} "
                f"tools={payload.get('tool_names')}"
            )
            return "\n".join(lines)
        if name == "llm_response":
            calls = payload.get("function_calls") or []
            if calls:
                return f"[llm <-] tool_calls={calls}"
            return (
                "[llm <-] "
                f"final={payload.get('final_preview')!r}"
            )
        if name == "model_action":
            return (
                f"[action] {payload.get('tool_name')} "
                f"{_compact_json(payload.get('arguments', {}))}"
            )
        if name == "policy_decision":
            return (
                f"[policy] {payload.get('tool_name')} "
                f"{payload.get('outcome')} - {payload.get('reason')}"
            )
        if name == "tool_started":
            return (
                f"[tool ->] {payload.get('tool_name')} "
                f"{_compact_json(payload.get('arguments', {}))}"
            )
        if name == "tool_finished":
            return (
                f"[tool <-] {payload.get('tool_name')} "
                f"success={payload.get('success')} summary={payload.get('summary')!r}"
            )
        if name == "run_finished":
            return f"[done] status={payload.get('status')}"
        return f"[trace] {name} {_compact_json(payload)}"


def preview(value: Any, max_chars: int = 500) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = _compact_json(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
