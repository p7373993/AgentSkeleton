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


class SessionStore:
    def __init__(self, logs_dir: Path) -> None:
        self.sessions_dir = logs_dir / "sessions"

    def load(self, name: str) -> SessionState:
        return SessionState(
            name=name,
            transcript=self._load_transcript(name),
        )

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

    def _dir_for(self, name: str) -> Path:
        return self.sessions_dir / self._safe_name(name)

    def _transcript_path_for(self, name: str) -> Path:
        return self._dir_for(name) / "transcript.jsonl"

    def _safe_name(self, name: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        return safe_name or "default"
