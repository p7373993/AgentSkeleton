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
        text = str(args["text"]).lower()
        domain = "finance" if "invoice" in text or "reconcile" in text else "general"
        return ToolResult(
            success=True,
            payload={"domain": domain},
            summary=f"classified {domain}",
        )


TOOLS = [ClassifyDomainTool()]
