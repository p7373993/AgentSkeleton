import json
from pathlib import Path

import pytest

from agentskeleton.core.session import SessionStore


def test_session_store_returns_empty_session_when_missing(tmp_path) -> None:
    session = SessionStore(tmp_path).load("default")

    assert session.name == "default"
    assert session.transcript == []


def test_session_store_appends_and_loads_transcript_turns(tmp_path) -> None:
    store = SessionStore(tmp_path)

    store.append_transcript("default", "user", "first")
    store.append_transcript("default", "assistant", "answer", {"run_id": "run-1"})
    session = store.load("default")

    assert [turn.role for turn in session.transcript] == ["user", "assistant"]
    assert [turn.content for turn in session.transcript] == ["first", "answer"]
    assert session.transcript[1].metadata == {"run_id": "run-1"}

    transcript_path = tmp_path / "sessions" / "default" / "transcript.jsonl"
    rows = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["role"] for row in rows] == ["user", "assistant"]


def test_session_store_serializes_non_json_metadata_values(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    marker = object()

    store.append_transcript(
        "default",
        "assistant",
        "answer",
        metadata={
            "workspace": tmp_path,
            "raw": marker,
            42: "numeric key",
        },
    )

    session = store.load("default")

    assert session.transcript[0].metadata["workspace"] == str(tmp_path)
    assert session.transcript[0].metadata["raw"] == str(marker)
    assert session.transcript[0].metadata["42"] == "numeric key"


def test_session_store_serializes_recursive_metadata_values(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    metadata = {}
    metadata["self"] = metadata

    store.append_transcript("default", "assistant", "answer", metadata=metadata)

    session = store.load("default")
    assert session.transcript[0].metadata == {"self": "<recursive>"}


def test_session_store_refreshes_summary_for_older_transcript_turns(tmp_path) -> None:
    store = SessionStore(tmp_path)
    store.append_transcript("default", "user", "old user")
    store.append_transcript("default", "assistant", "old answer")
    store.append_transcript("default", "user", "recent user")
    store.append_transcript("default", "assistant", "recent answer")

    store.refresh_summary("default", keep_turns=2)
    session = store.load("default")
    context = session.context_messages()

    assert session.summary == "- user: old user\n- assistant: old answer"
    assert [turn.content for turn in session.transcript] == [
        "old user",
        "old answer",
        "recent user",
        "recent answer",
    ]
    assert context[0].content == (
        "Prior conversation summary:\n"
        "- user: old user\n"
        "- assistant: old answer"
    )
    assert context[0].metadata == {
        "source": "session_summary",
        "sticky_context": True,
    }
    assert [turn.content for turn in context[1:]] == [
        "old user",
        "old answer",
        "recent user",
        "recent answer",
    ]


def test_session_store_limits_summary_to_recent_older_turns(tmp_path) -> None:
    store = SessionStore(tmp_path)
    store.append_transcript("default", "user", "very old user")
    store.append_transcript("default", "assistant", "very old answer")
    store.append_transcript("default", "user", "old user")
    store.append_transcript("default", "assistant", "old answer")
    store.append_transcript("default", "user", "recent user")
    store.append_transcript("default", "assistant", "recent answer")

    store.refresh_summary("default", keep_turns=2, summary_turns=2)

    assert store.load("default").summary == "- user: old user\n- assistant: old answer"


def test_session_store_lists_saved_sessions(tmp_path) -> None:
    store = SessionStore(tmp_path)
    store.append_transcript("default", "user", "hello")
    store.append_transcript("default", "assistant", "hi")
    store.append_transcript("work", "user", "old user")
    store.append_transcript("work", "assistant", "old answer")
    store.append_transcript("work", "user", "recent")
    store.refresh_summary("work", keep_turns=1)

    summaries = store.list_sessions()

    assert [summary.to_dict() for summary in summaries] == [
        {
            "name": "default",
            "transcript_turns": 2,
            "has_summary": False,
            "summary": None,
        },
        {
            "name": "work",
            "transcript_turns": 3,
            "has_summary": True,
            "summary": "- user: old user\n- assistant: old answer",
        },
    ]


def test_session_store_treats_sessions_root_file_as_empty(tmp_path) -> None:
    sessions_root = tmp_path / "sessions"
    sessions_root.write_text("not a directory", encoding="utf-8")

    assert SessionStore(tmp_path).list_sessions() == []


def test_session_store_reports_session_list_read_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    original_iterdir = Path.iterdir

    def fail_iterdir(path: Path):
        if path == sessions_root:
            raise OSError("permission denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)

    with pytest.raises(ValueError, match="Session list could not be read"):
        SessionStore(tmp_path).list_sessions()


def test_session_store_reports_session_list_root_stat_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    original_is_dir = Path.is_dir

    def fail_is_dir(path: Path) -> bool:
        if path == sessions_root:
            raise OSError("permission denied")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fail_is_dir)

    with pytest.raises(ValueError, match="Session list could not be read"):
        SessionStore(tmp_path).list_sessions()


def test_session_store_append_reports_session_path_file(tmp_path) -> None:
    session_dir = tmp_path / "sessions" / "default"
    session_dir.parent.mkdir()
    session_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="Session path is not a directory: default"):
        SessionStore(tmp_path).append_transcript("default", "user", "hello")


def test_session_store_refresh_reports_session_path_file(tmp_path) -> None:
    session_dir = tmp_path / "sessions" / "default"
    session_dir.parent.mkdir()
    session_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="Session path is not a directory: default"):
        SessionStore(tmp_path).refresh_summary("default", keep_turns=1)


def test_session_store_reports_session_dir_exists_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_dir = tmp_path / "sessions" / "default"
    original_exists = Path.exists

    def fail_exists(path: Path) -> bool:
        if path == session_dir:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_exists)

    with pytest.raises(ValueError, match="Session path could not be checked: default"):
        SessionStore(tmp_path).append_transcript("default", "user", "hello")


def test_session_store_reports_session_dir_type_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_dir = tmp_path / "sessions" / "default"
    session_dir.mkdir(parents=True)
    original_is_dir = Path.is_dir

    def fail_is_dir(path: Path) -> bool:
        if path == session_dir:
            raise OSError("permission denied")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fail_is_dir)

    with pytest.raises(ValueError, match="Session path could not be checked: default"):
        SessionStore(tmp_path).append_transcript("default", "user", "hello")


def test_session_store_reports_transcript_path_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = SessionStore(tmp_path)
    transcript_path = tmp_path / "sessions" / "default" / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True)
    original_exists = Path.exists

    def fail_exists(path: Path) -> bool:
        if path == transcript_path:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_exists)

    with pytest.raises(
        ValueError,
        match="Session transcript path could not be checked: default",
    ):
        store.append_transcript("default", "user", "hello")


def test_session_store_reports_summary_path_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = SessionStore(tmp_path)
    store.append_transcript("default", "user", "hello")
    summary_path = tmp_path / "sessions" / "default" / "summary.md"
    original_exists = Path.exists

    def fail_exists(path: Path) -> bool:
        if path == summary_path:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_exists)

    with pytest.raises(
        ValueError,
        match="Session summary path could not be checked: default",
    ):
        store.refresh_summary("default", keep_turns=1)


def test_session_store_ignores_malformed_transcript_lines(tmp_path) -> None:
    store = SessionStore(tmp_path)
    transcript_path = tmp_path / "sessions" / "default" / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "first"}),
                '{"role": "assistant"',
                json.dumps({"role": "assistant", "content": "second"}),
            ]
        ),
        encoding="utf-8",
    )

    session = store.load("default")

    assert [turn.content for turn in session.transcript] == ["first", "second"]
    assert store.list_sessions()[0].transcript_turns == 2


def test_session_store_ignores_transcript_directory(tmp_path) -> None:
    transcript_path = tmp_path / "sessions" / "default" / "transcript.jsonl"
    transcript_path.mkdir(parents=True)

    session = SessionStore(tmp_path).load("default")

    assert session.transcript == []


def test_session_store_ignores_invalid_utf8_transcript_lines(tmp_path) -> None:
    store = SessionStore(tmp_path)
    transcript_path = tmp_path / "sessions" / "default" / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_bytes(
        b'{"role": "user", "content": "first"}\n'
        b"\xff\xfe\x00broken\n"
        b'{"role": "assistant", "content": "second"}\n'
    )

    session = store.load("default")

    assert [turn.content for turn in session.transcript] == ["first", "second"]
    assert store.list_sessions()[0].transcript_turns == 2


def test_session_store_ignores_invalid_utf8_summary(tmp_path) -> None:
    store = SessionStore(tmp_path)
    summary_path = tmp_path / "sessions" / "default" / "summary.md"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_bytes(b"\xff\xfe\x00broken")

    session = store.load("default")

    assert session.summary is None
    assert store.list_sessions() == []


def test_session_store_reports_transcript_read_failures(
    tmp_path,
    monkeypatch,
) -> None:
    store = SessionStore(tmp_path)
    transcript_path = tmp_path / "sessions" / "default" / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(
        json.dumps({"role": "user", "content": "hello"}),
        encoding="utf-8",
    )
    original_read_bytes = Path.read_bytes

    def fail_read_bytes(path: Path) -> bytes:
        if path == transcript_path:
            raise OSError("permission denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    with pytest.raises(
        ValueError,
        match="Session transcript could not be read: default",
    ):
        store.load("default")


def test_session_store_reports_transcript_file_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = SessionStore(tmp_path)
    transcript_path = tmp_path / "sessions" / "default" / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(
        json.dumps({"role": "user", "content": "hello"}),
        encoding="utf-8",
    )
    original_is_file = Path.is_file

    def fail_is_file(path: Path) -> bool:
        if path == transcript_path:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_is_file)

    with pytest.raises(
        ValueError,
        match="Session transcript could not be read: default",
    ):
        store.load("default")


def test_session_store_reports_summary_read_failures(tmp_path, monkeypatch) -> None:
    store = SessionStore(tmp_path)
    summary_path = tmp_path / "sessions" / "default" / "summary.md"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text("summary", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs) -> str:
        if path == summary_path:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(
        ValueError,
        match="Session summary could not be read: default",
    ):
        store.load("default")


def test_session_store_reports_summary_file_stat_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = SessionStore(tmp_path)
    summary_path = tmp_path / "sessions" / "default" / "summary.md"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text("summary", encoding="utf-8")
    original_is_file = Path.is_file

    def fail_is_file(path: Path) -> bool:
        if path == summary_path:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_is_file)

    with pytest.raises(
        ValueError,
        match="Session summary could not be read: default",
    ):
        store.load("default")


def test_session_store_ignores_summary_directory(tmp_path) -> None:
    summary_path = tmp_path / "sessions" / "default" / "summary.md"
    summary_path.mkdir(parents=True)

    session = SessionStore(tmp_path).load("default")

    assert session.summary is None
    assert SessionStore(tmp_path).list_sessions() == []


def test_session_store_refresh_summary_ignores_malformed_transcript_lines(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    transcript_path = tmp_path / "sessions" / "default" / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "old"}),
                '{"role": "assistant"',
                json.dumps({"role": "assistant", "content": "recent"}),
            ]
        ),
        encoding="utf-8",
    )

    store.refresh_summary("default", keep_turns=1)

    assert store.load("default").summary == "- user: old"


def test_session_store_sanitizes_session_names(tmp_path) -> None:
    store = SessionStore(tmp_path)

    store.append_transcript("../bad name", "user", "safe")

    session = store.load("../bad name")
    assert [turn.content for turn in session.transcript] == ["safe"]
    assert not (tmp_path.parent / "bad name" / "transcript.jsonl").exists()


def test_session_store_ignores_legacy_previous_response_id_file(tmp_path) -> None:
    legacy_path = tmp_path / "sessions" / "default.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        json.dumps({"name": "default", "previous_response_id": "resp-legacy"}),
        encoding="utf-8",
    )

    session = SessionStore(tmp_path).load("default")

    assert session.transcript == []
