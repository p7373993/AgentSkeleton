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
        if not self.sessions_dir.exists():
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
        session_dir = self._dir_for(name)
        session_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        with self._transcript_path_for(name).open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

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
        session_dir = self._dir_for(name)
        session_dir.mkdir(parents=True, exist_ok=True)
        self._summary_path_for(name).write_text(summary, encoding="utf-8")

    def _load_transcript(self, name: str) -> list[ConversationMessage]:
        path = self._transcript_path_for(name)
        if not path.exists():
            return []

        transcript: list[ConversationMessage] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
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
        if not path.exists():
            return None
        summary = path.read_text(encoding="utf-8").strip()
        return summary or None

    def _dir_for(self, name: str) -> Path:
        return self.sessions_dir / self._safe_name(name)

    def _transcript_path_for(self, name: str) -> Path:
        return self._dir_for(name) / "transcript.jsonl"

    def _summary_path_for(self, name: str) -> Path:
        return self._dir_for(name) / "summary.md"

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
