from typing import Any, ClassVar

from agentskeleton.tools.base import Tool, ToolContext, ToolResult


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
        if context.ask_user is None:
            return ToolResult(
                success=False,
                summary="No user input callback configured",
                error="No user input callback configured",
            )

        question = str(args["question"])
        answer = context.ask_user(question)
        return ToolResult(
            success=True,
            payload={"question": question, "answer": answer},
            summary="User answered question",
        )
