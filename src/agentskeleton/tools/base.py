from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class ToolContext(BaseModel):
    workspace: Path
    logs_dir: Path = Path("runs")
    run_id: str | None = None
    shell_timeout_seconds: int = 30
    shell_max_output_bytes: int = 20000
    ask_user: Callable[[str], str] | None = None


class ToolResult(BaseModel):
    success: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    summary: str
    error: str | None = None


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    risk: ClassVar[str]
    args_schema: ClassVar[dict[str, Any]]

    @abstractmethod
    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute this tool with already-decoded arguments."""
