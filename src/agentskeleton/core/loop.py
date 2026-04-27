from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from agentskeleton.config import RunConfig
from agentskeleton.core.actions import (
    AgentAction,
    FinalAction,
    ToolCallAction,
    ToolCallBatchAction,
)
from agentskeleton.core.state import ConversationMessage, RunState, ToolObservation
from agentskeleton.core.trace import NullTraceSink, TraceSink
from agentskeleton.policy.permissions import PermissionDecision, PermissionPolicy
from agentskeleton.tools.base import ToolContext, ToolResult
from agentskeleton.tools.registry import ToolRegistry


class LLMLike(Protocol):
    def next_action(self, state: RunState, registry: ToolRegistry) -> AgentAction:
        """Return the next model-selected action."""


class LoggerLike(Protocol):
    path: object

    def log(self, event_type: str, step: int, payload: dict[str, object]) -> None:
        """Persist a run event."""


Confirmer = Callable[[PermissionDecision, ToolCallAction], bool]


class AgentLoop:
    def __init__(
        self,
        config: RunConfig,
        llm: LLMLike,
        registry: ToolRegistry,
        logger: LoggerLike,
        policy: PermissionPolicy | None = None,
        confirmer: Confirmer | None = None,
        ask_user: Callable[[str], str] | None = None,
        run_id: str | None = None,
        trace: TraceSink | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.registry = registry
        self.logger = logger
        self.policy = policy or PermissionPolicy(config.confirm_risky_actions)
        self.confirmer = confirmer or (lambda _decision, _action: False)
        self.ask_user = ask_user
        self.run_id = run_id
        self.trace = trace or NullTraceSink()

    def run(
        self,
        goal: str,
        conversation: list[ConversationMessage | dict[str, object]] | None = None,
        trace_context: dict[str, object] | None = None,
    ) -> RunState:
        state = RunState(
            run_id=self.run_id or str(uuid4()),
            workspace=self.config.workspace,
            goal=goal,
            conversation=self._normalize_conversation(conversation or []),
        )
        self.logger.log(
            "run_started",
            0,
            {"goal": goal, "workspace": str(state.workspace)},
        )
        self.trace.emit(
            "run_started",
            {
                "goal": goal,
                "workspace": str(state.workspace),
                "model": self.config.model,
                "resumed": bool(state.conversation),
                "conversation_turns": len(state.conversation),
                **(trace_context or {}),
            },
        )

        while state.step_count < self.config.max_steps:
            state.step_count += 1
            self.trace.emit("step_started", {"step": state.step_count})
            self.logger.log("model_requested", state.step_count, {"goal": goal})
            action = self.llm.next_action(state, self.registry)

            if isinstance(action, FinalAction):
                state.final_status = action.status
                state.final_answer = action.text
                self.trace.emit(
                    "run_finished",
                    {"status": state.final_status, "answer": action.text},
                )
                self.logger.log(
                    "run_finished",
                    state.step_count,
                    {"status": state.final_status, "answer": action.text},
                )
                return state

            if isinstance(action, ToolCallBatchAction):
                for tool_call in action.tool_calls:
                    self._execute_tool_action(state, tool_call)
                    if state.final_status is not None:
                        break
            else:
                self._execute_tool_action(state, action)
            if state.final_status is not None:
                self.trace.emit(
                    "run_finished",
                    {"status": state.final_status, "answer": state.final_answer},
                )
                self.logger.log(
                    "run_finished",
                    state.step_count,
                    {"status": state.final_status, "answer": state.final_answer},
                )
                return state

        state.final_status = "max_steps"
        self.trace.emit(
            "run_finished",
            {"status": state.final_status, "answer": state.final_answer},
        )
        self.logger.log(
            "run_finished",
            state.step_count,
            {"status": state.final_status, "answer": state.final_answer},
        )
        return state

    def _normalize_conversation(
        self,
        conversation: list[ConversationMessage | dict[str, object]],
    ) -> list[ConversationMessage]:
        normalized: list[ConversationMessage] = []
        for turn in conversation:
            if isinstance(turn, ConversationMessage):
                normalized.append(turn)
                continue
            metadata = turn.get("metadata") or {}
            normalized.append(
                ConversationMessage(
                    role=str(turn.get("role", "user")),
                    content=str(turn.get("content", "")),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        return normalized

    def _execute_tool_action(self, state: RunState, action: ToolCallAction) -> None:
        self.logger.log(
            "model_action",
            state.step_count,
            {
                "type": "tool_call",
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "call_id": action.call_id,
            },
        )
        self.trace.emit(
            "model_action",
            {
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "call_id": action.call_id,
            },
        )
        try:
            tool = self.registry.get(action.tool_name)
        except KeyError as exc:
            result = ToolResult(
                success=False,
                summary=str(exc),
                error="Unknown tool",
            )
            state.observations.append(
                ToolObservation(action.call_id, action.tool_name, "block", result)
            )
            state.final_status = "error"
            return

        decision = self.policy.decide(tool.name, action.arguments, tool.risk)
        self.logger.log(
            "policy_decision",
            state.step_count,
            {
                "tool_name": tool.name,
                "outcome": decision.outcome,
                "reason": decision.reason,
            },
        )
        self.trace.emit(
            "policy_decision",
            {
                "tool_name": tool.name,
                "outcome": decision.outcome,
                "reason": decision.reason,
            },
        )

        if decision.outcome == "block":
            result = ToolResult(
                success=False,
                summary=decision.reason,
                error="Permission blocked",
            )
            state.observations.append(
                ToolObservation(action.call_id, tool.name, decision.outcome, result)
            )
            state.final_status = "blocked"
            return

        if decision.outcome == "confirm" and not self.confirmer(decision, action):
            result = ToolResult(
                success=False,
                summary=decision.reason,
                error="Permission denied",
            )
            state.observations.append(
                ToolObservation(action.call_id, tool.name, decision.outcome, result)
            )
            state.final_status = "denied"
            return

        self.logger.log("tool_started", state.step_count, {"tool_name": tool.name})
        self.trace.emit(
            "tool_started",
            {"tool_name": tool.name, "arguments": action.arguments},
        )
        result = tool.execute(
            action.arguments,
            ToolContext(
                workspace=state.workspace,
                shell_timeout_seconds=self.config.shell_timeout_seconds,
                shell_max_output_bytes=self.config.shell_max_output_bytes,
                ask_user=self.ask_user,
            ),
        )
        state.observations.append(
            ToolObservation(action.call_id, tool.name, decision.outcome, result)
        )
        self.logger.log(
            "tool_finished",
            state.step_count,
            {
                "tool_name": tool.name,
                "success": result.success,
                "summary": result.summary,
                "error": result.error,
            },
        )
        self.trace.emit(
            "tool_finished",
            {
                "tool_name": tool.name,
                "success": result.success,
                "summary": result.summary,
                "error": result.error,
            },
        )
