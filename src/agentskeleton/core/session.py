import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentskeleton.core.state import ConversationMessage


@dataclass(frozen=True)
class SessionState:
    name: str
    transcript: list[ConversationMessage] = field(default_factory=list)
    summary: str | None = None

    def context_messages(self) -> list[ConversationMessage]:
        if not self.summary:
            return list(self.transcript)
        return [
            ConversationMessage(
                role="user",
                content=f"Prior conversation summary:\n{self.summary}",
                metadata={
                    "source": "session_summary",
                    "sticky_context": True,
                },
            ),
            *self.transcript,
        ]


@dataclass(frozen=True)
class SessionSummary:
    name: str
    transcript_turns: int
    summary: str | None = None

    @property
    def has_summary(self) -> bool:
        return self.summary is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "transcript_turns": self.transcript_turns,
            "has_summary": self.has_summary,
            "summary": self.summary,
        }


class SessionStore:
    def __init__(self, logs_dir: Path) -> None:
        self.sessions_dir = logs_dir / "sessions"

    def load(self, name: str) -> SessionState:
        return SessionState(
            name=name,
            transcript=self._load_transcript(name),
            summary=self._load_summary(name),
        )

    def list_sessions(self) -> list[SessionSummary]:
        if not self.sessions_dir.is_dir():
            return []
        summaries = []
        for session_dir in sorted(
            (item for item in self.sessions_dir.iterdir() if item.is_dir()),
            key=lambda item: item.name.lower(),
        ):
            name = session_dir.name
            transcript = self._load_transcript(name)
            summary = self._load_summary(name)
            if not transcript and summary is None:
                continue
            summaries.append(
                SessionSummary(
                    name=name,
                    transcript_turns=len(transcript),
                    summary=summary,
                )
            )
        return summaries

    def append_transcript(
        self,
        name: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_session_dir(name)
        row = {
            "role": role,
            "content": content,
            "metadata": _json_safe(metadata or {}),
        }
        transcript_path = self._transcript_path_for(name)
        safe_name = self._safe_name(name)
        if transcript_path.exists() and not transcript_path.is_file():
            raise ValueError(
                f"Session transcript path is not a file: {safe_name}"
            )
        try:
            with transcript_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise ValueError(
                f"Session transcript could not be written: {safe_name}"
            ) from exc

    def refresh_summary(
        self,
        name: str,
        keep_turns: int,
        summary_turns: int = 40,
    ) -> None:
        if keep_turns < 0:
            raise ValueError("keep_turns must be non-negative")
        if summary_turns <= 0:
            raise ValueError("summary_turns must be positive")

        transcript = self._load_transcript(name)
        older_turns = transcript[:-keep_turns] if keep_turns else transcript
        summary_turns_source = older_turns[-summary_turns:]
        summary = _summarize_transcript(summary_turns_source)
        self._ensure_session_dir(name)
        summary_path = self._summary_path_for(name)
        safe_name = self._safe_name(name)
        if summary_path.exists() and not summary_path.is_file():
            raise ValueError(f"Session summary path is not a file: {safe_name}")
        try:
            summary_path.write_text(summary, encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"Session summary could not be written: {safe_name}"
            ) from exc

    def _load_transcript(self, name: str) -> list[ConversationMessage]:
        path = self._transcript_path_for(name)
        if not path.is_file():
            return []

        transcript: list[ConversationMessage] = []
        safe_name = self._safe_name(name)
        try:
            raw_lines = path.read_bytes().splitlines()
        except OSError as exc:
            raise ValueError(
                f"Session transcript could not be read: {safe_name}"
            ) from exc
        for raw_line in raw_lines:
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata") or {}
            transcript.append(
                ConversationMessage(
                    role=str(row.get("role", "user")),
                    content=str(row.get("content", "")),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        return transcript

    def _load_summary(self, name: str) -> str | None:
        path = self._summary_path_for(name)
        if not path.is_file():
            return None
        safe_name = self._safe_name(name)
        try:
            summary = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(
                f"Session summary could not be read: {safe_name}"
            ) from exc
        except UnicodeDecodeError:
            return None
        return summary or None

    def _dir_for(self, name: str) -> Path:
        return self.sessions_dir / self._safe_name(name)

    def _transcript_path_for(self, name: str) -> Path:
        return self._dir_for(name) / "transcript.jsonl"

    def _summary_path_for(self, name: str) -> Path:
        return self._dir_for(name) / "summary.md"

    def _ensure_session_dir(self, name: str) -> Path:
        session_dir = self._dir_for(name)
        safe_name = self._safe_name(name)
        for candidate in (session_dir, *session_dir.parents):
            if not candidate.exists():
                continue
            if not candidate.is_dir():
                raise ValueError(f"Session path is not a directory: {safe_name}")
            break
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Session path could not be created: {safe_name}") from exc
        return session_dir

    def _safe_name(self, name: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        return safe_name or "default"


def _summarize_transcript(
    turns: list[ConversationMessage],
    max_turn_chars: int = 160,
) -> str:
    lines = []
    for turn in turns:
        content = re.sub(r"\s+", " ", turn.content).strip()
        if len(content) > max_turn_chars:
            content = f"{content[: max_turn_chars - 1]}..."
        lines.append(f"- {turn.role}: {content}")
    return "\n".join(lines)


def _json_safe(value: Any, seen: set[int] | None = None) -> Any:
    seen = seen or set()
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return {str(key): _json_safe(item, seen) for key, item in value.items()}
        finally:
            seen.remove(marker)
    if isinstance(value, list | tuple):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return [_json_safe(item, seen) for item in value]
        finally:
            seen.remove(marker)
    if value is None or isinstance(value, int | float | bool | str):
        return value
    return str(value)
