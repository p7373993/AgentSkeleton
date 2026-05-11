from typing import Any, ClassVar

from agentskeleton.tools.base import Tool, ToolContext, ToolResult


class ClassifyDomainTool(Tool):
    name: ClassVar[str] = "classify_domain"
    description: ClassVar[str] = "Classify a request into a broad work domain."
    risk: ClassVar[str] = "read"
    args_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        raw_text = args.get("text")
        if not isinstance(raw_text, str):
            return ToolResult(
                success=False,
                summary="Text invalid: text must be a string",
                error="Text invalid",
            )
        if not raw_text.strip():
            return ToolResult(
                success=False,
                summary="Text invalid: text cannot be blank",
                error="Text invalid",
            )
        text = raw_text.lower()
        domain = "finance" if "invoice" in text or "reconcile" in text else "general"
        return ToolResult(
            success=True,
            payload={"domain": domain},
            summary=f"classified {domain}",
        )


TOOLS = [ClassifyDomainTool()]
