import json
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
REPEATED_ACTION_THRESHOLD = 3


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
        self.policy = policy or PermissionPolicy(
            config.confirm_risky_actions,
            config.permission_profile,
        )
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
            try:
                action = self.llm.next_action(state, self.registry)
            except Exception as exc:
                state.final_status = "model_error"
                state.final_reason = f"Model call failed: {type(exc).__name__}"
                self.logger.log(
                    "run_error",
                    state.step_count,
                    {
                        "status": state.final_status,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                self.trace.emit(
                    "run_error",
                    {
                        "status": state.final_status,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                self._log_run_finished(state)
                return state

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

            if isinstance(action, ToolCallAction):
                self._execute_tool_action(state, action)
            elif isinstance(action, ToolCallBatchAction):
                if not action.tool_calls:
                    self._record_invalid_action(
                        state,
                        action,
                        reason="Model returned empty tool call batch",
                        error_type="empty_batch",
                    )
                    if state.final_status is not None:
                        self._log_run_finished(state)
                        return state
                for tool_call in action.tool_calls:
                    if not isinstance(tool_call, ToolCallAction):
                        self._record_invalid_action(state, tool_call)
                        break
                    self._execute_tool_action(state, tool_call)
                    if state.final_status is not None:
                        break
            else:
                self._record_invalid_action(state, action)
            if state.final_status is not None:
                self._log_run_finished(state)
                return state

        state.final_status = "max_steps"
        state.final_reason = f"Reached max_steps limit: {self.config.max_steps}"
        self._log_run_finished(state)
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
        if self._record_repeated_action(state, action):
            return

        try:
            tool = self.registry.get(action.tool_name)
        except KeyError as exc:
            summary = str(exc.args[0]) if exc.args else str(exc)
            result = ToolResult(
                success=False,
                payload={
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                },
                summary=summary,
                error="Unknown tool",
            )
            state.observations.append(
                ToolObservation(action.call_id, action.tool_name, "block", result)
            )
            state.final_status = "unknown_tool"
            state.final_reason = result.summary
            return

        validation_errors = _validate_tool_arguments(tool.args_schema, action.arguments)
        if validation_errors:
            result = ToolResult(
                success=False,
                payload={
                    "tool_name": tool.name,
                    "arguments": action.arguments,
                    "validation_errors": validation_errors,
                },
                summary="Invalid tool arguments",
                error="Invalid arguments",
            )
            state.observations.append(
                ToolObservation(action.call_id, tool.name, "block", result)
            )
            self._log_tool_finished(state, tool.name, result)
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
            state.final_reason = result.summary
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
            state.final_reason = result.summary
            return

        self.logger.log("tool_started", state.step_count, {"tool_name": tool.name})
        self.trace.emit(
            "tool_started",
            {"tool_name": tool.name, "arguments": action.arguments},
        )
        try:
            result = tool.execute(
                action.arguments,
                ToolContext(
                    workspace=state.workspace,
                    shell_timeout_seconds=self.config.shell_timeout_seconds,
                    shell_max_output_bytes=self.config.shell_max_output_bytes,
                    ask_user=self.ask_user,
                ),
            )
        except Exception as exc:
            result = ToolResult(
                success=False,
                payload={
                    "tool_name": tool.name,
                    "arguments": action.arguments,
                },
                summary=f"Tool raised an exception: {type(exc).__name__}",
                error=str(exc),
            )
            state.observations.append(
                ToolObservation(action.call_id, tool.name, decision.outcome, result)
            )
            self._log_tool_finished(state, tool.name, result)
            state.final_status = "tool_error"
            state.final_reason = result.summary
            return

        state.observations.append(
            ToolObservation(action.call_id, tool.name, decision.outcome, result)
        )
        self._log_tool_finished(state, tool.name, result)

    def _record_repeated_action(
        self,
        state: RunState,
        action: ToolCallAction,
    ) -> bool:
        fingerprint = _action_fingerprint(action)
        if fingerprint == state.last_action_fingerprint:
            state.repeated_action_count += 1
        else:
            state.last_action_fingerprint = fingerprint
            state.repeated_action_count = 1

        if state.repeated_action_count < REPEATED_ACTION_THRESHOLD:
            return False

        result = ToolResult(
            success=False,
            payload={
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "repeat_count": state.repeated_action_count,
            },
            summary=(
                f"Repeated tool action {state.repeated_action_count} times: "
                f"{action.tool_name}"
            ),
            error="Repeated action",
        )
        state.observations.append(
            ToolObservation(action.call_id, action.tool_name, "block", result)
        )
        state.final_status = "repeated_action"
        state.final_reason = result.summary
        return True

    def _log_tool_finished(
        self,
        state: RunState,
        tool_name: str,
        result: ToolResult,
    ) -> None:
        self.logger.log(
            "tool_finished",
            state.step_count,
            {
                "tool_name": tool_name,
                "success": result.success,
                "summary": result.summary,
                "error": result.error,
            },
        )
        self.trace.emit(
            "tool_finished",
            {
                "tool_name": tool_name,
                "success": result.success,
                "summary": result.summary,
                "error": result.error,
            },
        )

    def _record_invalid_action(
        self,
        state: RunState,
        action: object,
        reason: str | None = None,
        error_type: str | None = None,
    ) -> None:
        action_type = error_type or type(action).__name__
        reason = reason or f"Model returned unsupported action: {action_type}"
        state.final_status = "invalid_action"
        state.final_reason = reason
        payload = {
            "status": state.final_status,
            "error_type": action_type,
            "error": reason,
        }
        self.logger.log("run_error", state.step_count, payload)
        self.trace.emit("run_error", payload)

    def _log_run_finished(self, state: RunState) -> None:
        payload = {"status": state.final_status, "answer": state.final_answer}
        if state.final_reason:
            payload["reason"] = state.final_reason
        self.trace.emit("run_finished", payload)
        self.logger.log("run_finished", state.step_count, payload)


def _action_fingerprint(action: ToolCallAction) -> str:
    arguments = json.dumps(
        action.arguments,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{action.tool_name}:{arguments}"


def _validate_tool_arguments(
    schema: dict[str, object],
    arguments: dict[str, object],
) -> list[str]:
    if schema.get("type") != "object":
        return []

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    errors: list[str] = []
    required = schema.get("required", [])
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in arguments:
                errors.append(f"Missing required argument: {name}")

    if schema.get("additionalProperties") is False:
        for name in arguments:
            if name not in properties:
                errors.append(f"Unexpected argument: {name}")

    for name, value in arguments.items():
        property_schema = properties.get(name)
        if not isinstance(property_schema, dict):
            continue
        expected_type = property_schema.get("type")
        if isinstance(expected_type, str) and not _matches_json_type(
            value,
            expected_type,
        ):
            errors.append(f"Argument {name} must be {expected_type}")

    return errors


def _matches_json_type(value: object, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True
