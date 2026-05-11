from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentskeleton.tools.base import ToolResult


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ToolObservation:
    call_id: str
    tool_name: str
    policy_decision: str
    result: ToolResult


@dataclass
class RunState:
    run_id: str
    workspace: Path
    goal: str
    conversation: list[ConversationMessage] = field(default_factory=list)
    step_count: int = 0
    sent_observation_count: int = 0
    observations: list[ToolObservation] = field(default_factory=list)
    response_context_items: list[dict[str, Any]] = field(default_factory=list)
    last_action_fingerprint: str | None = None
    repeated_action_count: int = 0
    final_status: str | None = None
    final_answer: str | None = None
    final_reason: str | None = None
