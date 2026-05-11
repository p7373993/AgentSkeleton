from pathlib import Path

from agentskeleton.tools.base import ToolContext
from agentskeleton.tools.user import AskUserTool


def test_ask_user_tool_returns_callback_answer(tmp_path: Path) -> None:
    result = AskUserTool().execute(
        {"question": "Continue?"},
        ToolContext(
            workspace=tmp_path,
            ask_user=lambda question: f"answer to {question}",
        ),
    )

    assert result.success is True
    assert result.payload["answer"] == "answer to Continue?"


def test_ask_user_tool_fails_without_callback(tmp_path: Path) -> None:
    result = AskUserTool().execute(
        {"question": "Continue?"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "No user input callback configured"


def test_ask_user_tool_returns_error_when_callback_fails(tmp_path: Path) -> None:
    def failing_callback(question: str) -> str:
        raise RuntimeError("input channel closed")

    result = AskUserTool().execute(
        {"question": "Continue?"},
        ToolContext(workspace=tmp_path, ask_user=failing_callback),
    )

    assert result.success is False
    assert result.error == "User input failed"
    assert result.summary == "User input failed: RuntimeError"
    assert result.payload == {"question": "Continue?"}


def test_ask_user_tool_rejects_missing_question(tmp_path: Path) -> None:
    def fail_if_called(question: str) -> str:
        raise AssertionError("ask_user callback should not be called")

    result = AskUserTool().execute(
        {},
        ToolContext(workspace=tmp_path, ask_user=fail_if_called),
    )

    assert result.success is False
    assert result.error == "Question invalid"
    assert result.summary == "Question invalid: question must be a string"
    assert result.payload == {}


def test_ask_user_tool_rejects_non_string_question(tmp_path: Path) -> None:
    def fail_if_called(question: str) -> str:
        raise AssertionError("ask_user callback should not be called")

    result = AskUserTool().execute(
        {"question": 123},
        ToolContext(workspace=tmp_path, ask_user=fail_if_called),
    )

    assert result.success is False
    assert result.error == "Question invalid"
    assert result.summary == "Question invalid: question must be a string"
    assert result.payload == {}


def test_ask_user_tool_rejects_blank_question(tmp_path: Path) -> None:
    def fail_if_called(question: str) -> str:
        raise AssertionError("ask_user callback should not be called")

    result = AskUserTool().execute(
        {"question": "   "},
        ToolContext(workspace=tmp_path, ask_user=fail_if_called),
    )

    assert result.success is False
    assert result.error == "Question invalid"
    assert result.summary == "Question invalid: question cannot be blank"
    assert result.payload == {}
