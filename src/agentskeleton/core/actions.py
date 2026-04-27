from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCallAction:
    tool_name: str
    arguments: dict[str, Any]
    call_id: str
    provider_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCallBatchAction:
    tool_calls: list[ToolCallAction]


@dataclass(frozen=True)
class FinalAction:
    text: str
    status: str = "completed"


AgentAction = ToolCallAction | ToolCallBatchAction | FinalAction
