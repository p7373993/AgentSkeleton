import json

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
