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
            return _text_error("text must be a string")
        stripped_text = _safe_strip(raw_text)
        if stripped_text is None:
            return _text_error("text could not be inspected")
        if not stripped_text:
            return _text_error("text cannot be blank")
        text = _safe_lower(raw_text)
        if text is None:
            return _text_error("text could not be inspected")
        domain = _classify_domain(text)
        return ToolResult(
            success=True,
            payload={"domain": domain},
            summary=f"classified {domain}",
        )


def _text_error(message: str) -> ToolResult:
    return ToolResult(
        success=False,
        summary=f"Text invalid: {message}",
        error="Text invalid",
    )


def _safe_strip(value: str) -> str | None:
    try:
        stripped = value.strip()
    except Exception:
        return None
    return str.__str__(stripped)


def _safe_lower(value: str) -> str | None:
    try:
        lowered = value.lower()
    except Exception:
        return None
    return str.__str__(lowered)


def _classify_domain(text: str) -> str:
    for domain, keywords in DOMAIN_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return domain
    return "general"


TOOLS = [ClassifyDomainTool()]
