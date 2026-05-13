from typing import Any, ClassVar

from agentskeleton.tools.base import Tool, ToolContext, ToolResult


def _safe_strip(value: str) -> str | None:
    try:
        return value.strip()
    except Exception:
        return None


class AskUserTool(Tool):
    name: ClassVar[str] = "ask_user"
    description: ClassVar[str] = "Ask the user one direct question."
    risk: ClassVar[str] = "interactive"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A direct question to ask the user.",
            }
        },
        "required": ["question"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        raw_question = args.get("question")
        if not isinstance(raw_question, str):
            return ToolResult(
                success=False,
                summary="Question invalid: question must be a string",
                error="Question invalid",
            )
        question = raw_question
        stripped_question = _safe_strip(question)
        if stripped_question is None:
            return ToolResult(
                success=False,
                summary="Question invalid: question could not be inspected",
                error="Question invalid",
            )
        if not stripped_question:
            return ToolResult(
                success=False,
                summary="Question invalid: question cannot be blank",
                error="Question invalid",
            )

        if context.ask_user is None:
            return ToolResult(
                success=False,
                summary="No user input callback configured",
                error="No user input callback configured",
            )

        try:
            answer = context.ask_user(question)
        except Exception as exc:
            return ToolResult(
                success=False,
                payload={"question": question},
                summary=f"User input failed: {type(exc).__name__}",
                error="User input failed",
            )
        if not isinstance(answer, str):
            return ToolResult(
                success=False,
                payload={"question": question},
                summary=f"User input returned invalid answer: {type(answer).__name__}",
                error="User input invalid",
            )
        return ToolResult(
            success=True,
            payload={"question": question, "answer": answer},
            summary="User answered question",
        )
