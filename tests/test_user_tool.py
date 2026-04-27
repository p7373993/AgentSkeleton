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
