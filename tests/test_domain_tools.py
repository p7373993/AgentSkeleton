from pathlib import Path

import pytest

from agentskeleton.domain_tools import ClassifyDomainTool
from agentskeleton.tools.base import ToolContext


def test_classify_domain_detects_finance_terms(tmp_path: Path) -> None:
    result = ClassifyDomainTool().execute(
        {"text": "Please reconcile this invoice"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["domain"] == "finance"
    assert result.summary == "classified finance"


def test_classify_domain_defaults_to_general(tmp_path: Path) -> None:
    result = ClassifyDomainTool().execute(
        {"text": "Write a short product brief"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["domain"] == "general"
    assert result.summary == "classified general"


@pytest.mark.parametrize(
    ("args", "summary"),
    [
        ({}, "Text invalid: text must be a string"),
        ({"text": 123}, "Text invalid: text must be a string"),
        ({"text": "   "}, "Text invalid: text cannot be blank"),
    ],
)
def test_classify_domain_rejects_invalid_text(
    tmp_path: Path,
    args: dict[str, object],
    summary: str,
) -> None:
    result = ClassifyDomainTool().execute(args, ToolContext(workspace=tmp_path))

    assert result.success is False
    assert result.error == "Text invalid"
    assert result.summary == summary
    assert result.payload == {}
