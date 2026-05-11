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
