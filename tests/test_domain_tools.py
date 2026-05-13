from pathlib import Path

import pytest

from agentskeleton.domain_tools import ClassifyDomainTool
from agentskeleton.tools.base import ToolContext


class UninspectableString(str):
    def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("cannot strip")


class UnlowerableString(str):
    def lower(self) -> str:
        raise RuntimeError("cannot lower")


def test_classify_domain_detects_finance_terms(tmp_path: Path) -> None:
    result = ClassifyDomainTool().execute(
        {"text": "Please reconcile this invoice"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["domain"] == "finance"
    assert result.summary == "classified finance"


def test_classify_domain_normalizes_text_subclass_lower_output(
    tmp_path: Path,
) -> None:
    class UnsearchableString(str):
        def __str__(self) -> str:
            return self

        def lower(self) -> str:
            return self

        def __contains__(self, item: object) -> bool:
            raise RuntimeError("text search unavailable")

    result = ClassifyDomainTool().execute(
        {"text": UnsearchableString("Please reconcile this invoice")},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["domain"] == "finance"
    assert result.summary == "classified finance"


def test_classify_domain_defaults_to_general(tmp_path: Path) -> None:
    result = ClassifyDomainTool().execute(
        {"text": "Say hello"},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["domain"] == "general"
    assert result.summary == "classified general"


@pytest.mark.parametrize(
    ("text", "domain"),
    [
        ("Fix this Python test failure", "coding"),
        ("Summarize this CSV dataset", "data"),
        ("Read the README file", "filesystem"),
        ("Create an output artifact", "artifacts"),
        ("Ask the user for clarification", "interactive"),
        ("Load a custom tool module", "tool_packs"),
        ("Write a short product brief", "writing"),
    ],
)
def test_classify_domain_detects_broad_work_domains(
    tmp_path: Path,
    text: str,
    domain: str,
) -> None:
    result = ClassifyDomainTool().execute(
        {"text": text},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.payload["domain"] == domain
    assert result.summary == f"classified {domain}"


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


@pytest.mark.parametrize(
    "text",
    [
        UninspectableString("Please reconcile this invoice"),
        UnlowerableString("Please reconcile this invoice"),
    ],
)
def test_classify_domain_rejects_uninspectable_text(
    tmp_path: Path,
    text: str,
) -> None:
    result = ClassifyDomainTool().execute(
        {"text": text},
        ToolContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.error == "Text invalid"
    assert result.summary == "Text invalid: text could not be inspected"
    assert result.payload == {}
