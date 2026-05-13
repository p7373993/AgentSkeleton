import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentskeleton.core.state import ConversationMessage

_MAX_SESSION_DIR_NAME_LENGTH = 120
_MAX_SESSION_FILE_BYTES = 2_097_152
_MAX_TRANSCRIPT_CONTENT_CHARS = 4_096
_MAX_TRANSCRIPT_METADATA_VALUE_CHARS = 4_096
_MAX_TRANSCRIPT_METADATA_PREVIEW_CHARS = 200
_MAX_TRANSCRIPT_METADATA_DEPTH = 64
_MAX_TRANSCRIPT_METADATA_ITEMS = 200
_MAX_DEPTH_EXCEEDED = "<max-depth-exceeded>"
_UNINSPECTABLE_VALUE = "<uninspectable>"
_TRUNCATED_ITEMS_KEY = "__truncated_items__"
_WINDOWS_RESERVED_SESSION_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class SessionState:
    name: str
    transcript: list[ConversationMessage] = field(default_factory=list)
    summary: str | None = None

    def context_messages(self) -> list[ConversationMessage]:
        if self.summary is None:
            return list(self.transcript)
        return [
            ConversationMessage(
                role="user",
                content=f"Prior conversation summary:\n{_safe_text(self.summary)}",
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
        try:
            sessions_root_is_directory = self.sessions_dir.is_dir()
        except OSError as exc:
            raise ValueError(
                f"Session list could not be read: {self.sessions_dir}"
            ) from exc
        if not sessions_root_is_directory:
            return []
        try:
            session_dirs = [
                item for item in self.sessions_dir.iterdir() if item.is_dir()
            ]
        except OSError as exc:
            raise ValueError(
                f"Session list could not be read: {self.sessions_dir}"
            ) from exc
        summaries = []
        for session_dir in sorted(
            session_dirs,
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
        role: object,
        content: object,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_session_dir(name)
        row = {
            "role": _safe_text(role),
            "content": _bounded_transcript_content(_safe_text(content)),
            "metadata": _json_safe(metadata if metadata is not None else {}),
        }
        transcript_path = self._transcript_path_for(name)
        safe_name = self._safe_name(name)
        try:
            transcript_exists = transcript_path.exists()
        except OSError as exc:
            raise ValueError(
                f"Session transcript path could not be checked: {safe_name}"
            ) from exc
        if transcript_exists:
            try:
                transcript_is_file = transcript_path.is_file()
            except OSError as exc:
                raise ValueError(
                    f"Session transcript path could not be checked: {safe_name}"
                ) from exc
            if not transcript_is_file:
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
        try:
            summary_exists = summary_path.exists()
        except OSError as exc:
            raise ValueError(
                f"Session summary path could not be checked: {safe_name}"
            ) from exc
        if summary_exists:
            try:
                summary_is_file = summary_path.is_file()
            except OSError as exc:
                raise ValueError(
                    f"Session summary path could not be checked: {safe_name}"
                ) from exc
            if not summary_is_file:
                raise ValueError(f"Session summary path is not a file: {safe_name}")
        try:
            summary_path.write_text(summary, encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"Session summary could not be written: {safe_name}"
            ) from exc

    def _load_transcript(self, name: str) -> list[ConversationMessage]:
        path = self._transcript_path_for(name)
        safe_name = self._safe_name(name)
        try:
            transcript_is_file = path.is_file()
        except OSError as exc:
            raise ValueError(
                f"Session transcript could not be read: {safe_name}"
            ) from exc
        if not transcript_is_file:
            return []

        transcript: list[ConversationMessage] = []
        try:
            if path.stat().st_size > _MAX_SESSION_FILE_BYTES:
                raise ValueError(
                    f"Session transcript exceeds {_MAX_SESSION_FILE_BYTES} bytes: "
                    f"{safe_name}"
                )
            raw_transcript = path.read_bytes()
            if len(raw_transcript) > _MAX_SESSION_FILE_BYTES:
                raise ValueError(
                    f"Session transcript exceeds {_MAX_SESSION_FILE_BYTES} bytes: "
                    f"{safe_name}"
                )
            raw_lines = raw_transcript.splitlines()
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
            except (json.JSONDecodeError, RecursionError):
                continue
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata") or {}
            safe_metadata = _json_safe(metadata) if isinstance(metadata, dict) else {}
            transcript.append(
                ConversationMessage(
                    role=_safe_text(row.get("role", "user")),
                    content=_bounded_transcript_content(
                        _safe_text(row.get("content", "")),
                    ),
                    metadata=safe_metadata
                    if isinstance(safe_metadata, dict)
                    else {},
                )
            )
        return transcript

    def _load_summary(self, name: str) -> str | None:
        path = self._summary_path_for(name)
        safe_name = self._safe_name(name)
        try:
            summary_is_file = path.is_file()
        except OSError as exc:
            raise ValueError(
                f"Session summary could not be read: {safe_name}"
            ) from exc
        if not summary_is_file:
            return None
        try:
            if path.stat().st_size > _MAX_SESSION_FILE_BYTES:
                raise ValueError(
                    f"Session summary exceeds {_MAX_SESSION_FILE_BYTES} bytes: "
                    f"{safe_name}"
                )
            raw_summary = path.read_text(encoding="utf-8")
            raw_summary_bytes = _utf8_size(raw_summary)
            if raw_summary_bytes is None:
                raise ValueError(
                    f"Session summary could not be inspected: {safe_name}"
                )
            if raw_summary_bytes > _MAX_SESSION_FILE_BYTES:
                raise ValueError(
                    f"Session summary exceeds {_MAX_SESSION_FILE_BYTES} bytes: "
                    f"{safe_name}"
                )
            summary = _bounded_transcript_content(
                raw_summary.strip(),
            )
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
            try:
                candidate_exists = candidate.exists()
            except OSError as exc:
                raise ValueError(
                    f"Session path could not be checked: {safe_name}"
                ) from exc
            if not candidate_exists:
                continue
            try:
                candidate_is_directory = candidate.is_dir()
            except OSError as exc:
                raise ValueError(
                    f"Session path could not be checked: {safe_name}"
                ) from exc
            if not candidate_is_directory:
                raise ValueError(f"Session path is not a directory: {safe_name}")
            break
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Session path could not be created: {safe_name}") from exc
        return session_dir

    def _safe_name(self, name: str) -> str:
        name_text = _safe_text(name)
        if name_text == _UNINSPECTABLE_VALUE:
            name_text = "default"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name_text).strip("._")
        safe_name = safe_name or "default"
        if len(safe_name) > _MAX_SESSION_DIR_NAME_LENGTH:
            digest = hashlib.sha256(safe_name.encode("utf-8")).hexdigest()[:12]
            prefix_length = _MAX_SESSION_DIR_NAME_LENGTH - len(digest) - 1
            safe_name = f"{safe_name[:prefix_length]}-{digest}"
        base_name = safe_name.split(".", 1)[0].upper()
        if base_name in _WINDOWS_RESERVED_SESSION_BASENAMES:
            return f"{safe_name}_"
        return safe_name


def _summarize_transcript(
    turns: list[ConversationMessage],
    max_turn_chars: int = 160,
) -> str:
    lines = []
    for turn in turns:
        content = re.sub(r"\s+", " ", _safe_text(turn.content)).strip()
        if len(content) > max_turn_chars:
            content = f"{content[: max_turn_chars - 1]}..."
        lines.append(f"- {_safe_text(turn.role)}: {content}")
    return "\n".join(lines)


def _bounded_transcript_content(content: str) -> str:
    if len(content) <= _MAX_TRANSCRIPT_CONTENT_CHARS:
        return content
    omitted = len(content) - _MAX_TRANSCRIPT_CONTENT_CHARS
    return (
        f"{content[:_MAX_TRANSCRIPT_CONTENT_CHARS]}"
        f"\n[truncated {omitted} characters]"
    )


def _safe_text(value: object) -> str:
    try:
        text = str(value)
    except Exception:
        return _UNINSPECTABLE_VALUE
    return str.__str__(text)


def _utf8_size(value: str) -> int | None:
    try:
        return len(value.encode("utf-8"))
    except Exception:
        return None


def _json_safe(value: Any, seen: set[int] | None = None, depth: int = 0) -> Any:
    if depth > _MAX_TRANSCRIPT_METADATA_DEPTH:
        return _MAX_DEPTH_EXCEEDED
    seen = seen or set()
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return _json_safe_mapping(value, seen, depth)
        finally:
            seen.remove(marker)
    if isinstance(value, list | tuple):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return _json_safe_sequence(value, seen, depth)
        finally:
            seen.remove(marker)
    if isinstance(value, str):
        return _bounded_metadata_string(value)
    if value is None or isinstance(value, int | float | bool):
        return value
    return _bounded_metadata_string(_safe_text(value))


def _json_safe_mapping(
    value: dict[Any, Any],
    seen: set[int],
    depth: int,
) -> dict[str, Any] | str:
    safe_items: dict[str, Any] = {}
    pending_key = ""
    pending_item: Any = None
    has_pending_item = False
    omitted_after_pending = 0
    total_items = 0
    try:
        for index, (key, item) in enumerate(value.items()):
            total_items = index + 1
            safe_key = _safe_text(key)
            safe_item = _json_safe(item, seen, depth + 1)
            if index < _MAX_TRANSCRIPT_METADATA_ITEMS - 1:
                safe_items[safe_key] = safe_item
                continue
            if index == _MAX_TRANSCRIPT_METADATA_ITEMS - 1:
                pending_key = safe_key
                pending_item = safe_item
                has_pending_item = True
                continue
            omitted_after_pending += 1
    except Exception:
        return _UNINSPECTABLE_VALUE
    if omitted_after_pending:
        omitted = omitted_after_pending + int(has_pending_item)
        safe_items[_TRUNCATED_ITEMS_KEY] = _truncated_items_marker(
            total_items,
            omitted,
        )
    elif has_pending_item:
        safe_items[pending_key] = pending_item
    return safe_items


def _json_safe_sequence(
    value: list[Any] | tuple[Any, ...],
    seen: set[int],
    depth: int,
) -> list[Any] | str:
    safe_items: list[Any] = []
    pending_item: Any = None
    has_pending_item = False
    omitted_after_pending = 0
    total_items = 0
    try:
        for index, item in enumerate(value):
            total_items = index + 1
            safe_item = _json_safe(item, seen, depth + 1)
            if index < _MAX_TRANSCRIPT_METADATA_ITEMS - 1:
                safe_items.append(safe_item)
                continue
            if index == _MAX_TRANSCRIPT_METADATA_ITEMS - 1:
                pending_item = safe_item
                has_pending_item = True
                continue
            omitted_after_pending += 1
    except Exception:
        return _UNINSPECTABLE_VALUE
    if omitted_after_pending:
        omitted = omitted_after_pending + int(has_pending_item)
        safe_items.append(_truncated_items_marker(total_items, omitted))
    elif has_pending_item:
        safe_items.append(pending_item)
    return safe_items


def _truncated_items_marker(total_items: int, omitted: int) -> dict[str, object]:
    return {
        "truncated": True,
        "items": total_items,
        "omitted": omitted,
    }


def _bounded_metadata_string(value: str) -> str | dict[str, object]:
    encoded_size = _utf8_size(value)
    if encoded_size is None:
        return _UNINSPECTABLE_VALUE
    value = str.__str__(value)
    if len(value) <= _MAX_TRANSCRIPT_METADATA_VALUE_CHARS:
        return value
    return {
        "truncated": True,
        "bytes": encoded_size,
        "preview": f"{value[:_MAX_TRANSCRIPT_METADATA_PREVIEW_CHARS]}...",
    }
