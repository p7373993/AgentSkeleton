from typing import Any, ClassVar

from agentskeleton.tools.base import Tool, ToolContext, ToolResult

DOMAIN_KEYWORDS = (
    ("finance", ("invoice", "reconcile", "q4", "ledger", "expense")),
    (
        "coding",
        ("bug", "code", "fix", "python", "test failure", "compile", "refactor"),
    ),
    ("data", ("csv", "dataset", "spreadsheet", "table", "chart", "analyze")),
    ("filesystem", ("readme", "file", "directory", "folder", "path")),
    ("artifacts", ("artifact", "document", "report", "output")),
    ("interactive", ("ask the user", "clarification", "confirm", "question")),
    ("tool_packs", ("tool module", "tool pack", "custom tool", "plugin")),
    ("writing", ("write", "draft", "brief", "copy", "summary")),
)


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
        domain = _classify_domain(text)
        return ToolResult(
            success=True,
            payload={"domain": domain},
            summary=f"classified {domain}",
        )


def _classify_domain(text: str) -> str:
    for domain, keywords in DOMAIN_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return domain
    return "general"


TOOLS = [ClassifyDomainTool()]
