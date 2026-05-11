import hashlib
import json
from collections.abc import Callable, Mapping
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
MAX_TOOL_CALL_BATCH_SIZE = 20
MAX_LOGGED_ARGUMENT_BYTES = 4_096
MAX_LOGGED_ARGUMENT_PREVIEW_CHARS = 49
MAX_LOGGED_TEXT_BYTES = 4_096
MAX_LOGGED_TEXT_PREVIEW_CHARS = 200
MAX_LOGGED_VALUE_DEPTH = 64
MAX_DEPTH_EXCEEDED = "<max-depth-exceeded>"
MAX_TOOL_ACTION_METADATA_BYTES = 512


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
        trace_context: object | None = None,
    ) -> RunState:
        normalized_goal = str(goal)
        state = RunState(
            run_id=self.run_id or str(uuid4()),
            workspace=self.config.workspace,
            goal=normalized_goal,
            conversation=self._normalize_conversation(conversation),
        )
        start_payload = {
            **_trace_context(trace_context),
            "goal": _logged_text(normalized_goal),
            "workspace": str(state.workspace),
            "resumed": bool(state.conversation),
            "conversation_turns": len(state.conversation),
        }
        self._log_event(
            "run_started",
            0,
            start_payload,
        )
        self._emit_trace(
            "run_started",
            {
                "model": self.config.model,
                **start_payload,
            },
        )
        if not normalized_goal.strip():
            state.final_status = "invalid_goal"
            state.final_reason = "Goal cannot be blank"
            self._log_run_finished(state)
            return state

        while state.step_count < self.config.max_steps:
            state.step_count += 1
            self._emit_trace("step_started", {"step": state.step_count})
            self._log_event(
                "model_requested",
                state.step_count,
                {"goal": _logged_text(normalized_goal)},
            )
            try:
                action = self.llm.next_action(state, self.registry)
            except Exception as exc:
                state.final_status = "model_error"
                state.final_reason = f"Model call failed: {type(exc).__name__}"
                error_payload = {
                    "status": state.final_status,
                    "error_type": type(exc).__name__,
                    "error": _logged_text(str(exc)),
                }
                self._log_event("run_error", state.step_count, error_payload)
                self._emit_trace("run_error", error_payload)
                self._log_run_finished(state)
                return state

            if isinstance(action, FinalAction):
                metadata_error = _validate_final_action_metadata(action)
                if metadata_error is not None:
                    self._record_invalid_action(
                        state,
                        action,
                        reason=f"Model returned invalid final action: {metadata_error}",
                        error_type="invalid_final_action",
                    )
                    self._log_run_finished(state)
                    return state
                state.final_status = action.status
                state.final_answer = action.text
                self._log_run_finished(state)
                return state

            if isinstance(action, ToolCallAction):
                self._execute_tool_action(state, action)
            elif isinstance(action, ToolCallBatchAction):
                if not isinstance(action.tool_calls, list):
                    self._record_invalid_action(
                        state,
                        action,
                        reason=(
                            "Model returned invalid tool call batch: "
                            "tool_calls must be a list"
                        ),
                        error_type="invalid_tool_batch",
                    )
                    if state.final_status is not None:
                        self._log_run_finished(state)
                        return state
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
                if len(action.tool_calls) > MAX_TOOL_CALL_BATCH_SIZE:
                    self._record_invalid_action(
                        state,
                        action,
                        reason=(
                            "Model returned invalid tool call batch: "
                            "too many tool calls (max "
                            f"{MAX_TOOL_CALL_BATCH_SIZE})"
                        ),
                        error_type="invalid_tool_batch",
                    )
                    if state.final_status is not None:
                        self._log_run_finished(state)
                        return state
                for tool_call in action.tool_calls:
                    if not isinstance(tool_call, ToolCallAction):
                        self._record_invalid_action(state, tool_call)
                        if state.final_status is not None:
                            self._log_run_finished(state)
                            return state
                        continue
                    metadata_error = _validate_tool_action_metadata(tool_call)
                    if metadata_error is not None:
                        self._record_invalid_action(
                            state,
                            tool_call,
                            reason=(
                                "Model returned invalid tool call: "
                                f"{metadata_error}"
                            ),
                            error_type="invalid_tool_call",
                        )
                        if state.final_status is not None:
                            self._log_run_finished(state)
                            return state
                duplicate_call_id = _duplicate_batch_call_id(action.tool_calls)
                if duplicate_call_id is not None:
                    self._record_invalid_action(
                        state,
                        action,
                        reason=(
                            "Model returned invalid tool call batch: "
                            f"duplicate call_id: {duplicate_call_id}"
                        ),
                        error_type="invalid_tool_batch",
                    )
                    if state.final_status is not None:
                        self._log_run_finished(state)
                        return state
                for tool_call in action.tool_calls:
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
        conversation: object | None,
    ) -> list[ConversationMessage]:
        if conversation is None:
            return []
        turns = (
            conversation
            if isinstance(conversation, (list, tuple))
            else [conversation]
        )
        normalized: list[ConversationMessage] = []
        for turn in turns:
            if isinstance(turn, ConversationMessage):
                normalized.append(
                    _conversation_message(
                        turn.role,
                        turn.content,
                        turn.metadata,
                    )
                )
                continue
            if not isinstance(turn, Mapping):
                normalized.append(
                    ConversationMessage(role="user", content=str(turn)),
                )
                continue
            normalized.append(
                _conversation_message(
                    turn.get("role", "user"),
                    turn.get("content", ""),
                    turn.get("metadata") or {},
                )
            )
        return normalized

    def _execute_tool_action(self, state: RunState, action: ToolCallAction) -> None:
        metadata_error = _validate_tool_action_metadata(action)
        if metadata_error is not None:
            self._record_invalid_action(
                state,
                action,
                reason=f"Model returned invalid tool call: {metadata_error}",
                error_type="invalid_tool_call",
            )
            return

        logged_arguments = _logged_arguments(action.arguments)
        self._log_event(
            "model_action",
            state.step_count,
            {
                "type": "tool_call",
                "tool_name": action.tool_name,
                "arguments": logged_arguments,
                "call_id": action.call_id,
            },
        )
        self._emit_trace(
            "model_action",
            {
                "tool_name": action.tool_name,
                "arguments": logged_arguments,
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
            self._log_tool_finished(state, action.tool_name, result)
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
        self._log_event(
            "policy_decision",
            state.step_count,
            {
                "tool_name": tool.name,
                "outcome": decision.outcome,
                "reason": decision.reason,
            },
        )
        self._emit_trace(
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
            self._log_tool_finished(state, tool.name, result)
            state.final_status = "blocked"
            state.final_reason = result.summary
            return

        if decision.outcome == "confirm":
            try:
                confirmed = self.confirmer(decision, action)
            except Exception as exc:
                result = ToolResult(
                    success=False,
                    summary=f"Permission confirmation failed: {type(exc).__name__}",
                    error=str(exc),
                )
                state.observations.append(
                    ToolObservation(
                        action.call_id,
                        tool.name,
                        decision.outcome,
                        result,
                    )
                )
                self._log_tool_finished(state, tool.name, result)
                state.final_status = "denied"
                state.final_reason = result.summary
                return
            if not isinstance(confirmed, bool):
                result = ToolResult(
                    success=False,
                    summary=(
                        "Permission confirmation returned invalid result: "
                        f"{type(confirmed).__name__}"
                    ),
                    error="Permission confirmation invalid",
                )
                state.observations.append(
                    ToolObservation(
                        action.call_id,
                        tool.name,
                        decision.outcome,
                        result,
                    )
                )
                self._log_tool_finished(state, tool.name, result)
                state.final_status = "denied"
                state.final_reason = result.summary
                return
            if not confirmed:
                result = ToolResult(
                    success=False,
                    summary=decision.reason,
                    error="Permission denied",
                )
                state.observations.append(
                    ToolObservation(
                        action.call_id,
                        tool.name,
                        decision.outcome,
                        result,
                    )
                )
                self._log_tool_finished(state, tool.name, result)
                state.final_status = "denied"
                state.final_reason = result.summary
                return

        self._log_event("tool_started", state.step_count, {"tool_name": tool.name})
        self._emit_trace(
            "tool_started",
            {"tool_name": tool.name, "arguments": logged_arguments},
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

        if not isinstance(result, ToolResult):
            result = ToolResult(
                success=False,
                payload={
                    "tool_name": tool.name,
                    "arguments": action.arguments,
                    "result_type": type(result).__name__,
                },
                summary=f"Tool returned invalid result: {type(result).__name__}",
                error="Invalid tool result",
            )
            state.observations.append(
                ToolObservation(action.call_id, tool.name, decision.outcome, result)
            )
            self._log_tool_finished(state, tool.name, result)
            state.final_status = "tool_error"
            state.final_reason = result.summary
            return

        result_errors = _validate_tool_result_metadata(result)
        if result_errors:
            result = ToolResult(
                success=False,
                payload={
                    "tool_name": tool.name,
                    "arguments": action.arguments,
                    "validation_errors": result_errors,
                },
                summary=f"Tool returned malformed result: {result_errors[0]}",
                error="Malformed tool result",
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
        self._log_tool_finished(state, action.tool_name, result)
        state.final_status = "repeated_action"
        state.final_reason = result.summary
        return True

    def _log_tool_finished(
        self,
        state: RunState,
        tool_name: str,
        result: ToolResult,
    ) -> None:
        summary = _logged_text(result.summary)
        error = _logged_text(result.error) if result.error is not None else None
        self._log_event(
            "tool_finished",
            state.step_count,
            {
                "tool_name": tool_name,
                "success": result.success,
                "summary": summary,
                "error": error,
            },
        )
        self._emit_trace(
            "tool_finished",
            {
                "tool_name": tool_name,
                "success": result.success,
                "summary": summary,
                "error": error,
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
        self._log_event("run_error", state.step_count, payload)
        self._emit_trace("run_error", payload)

    def _log_run_finished(self, state: RunState) -> None:
        payload = {
            "status": state.final_status,
            "answer": (
                _logged_text(state.final_answer)
                if state.final_answer is not None
                else None
            ),
        }
        if state.final_reason:
            payload["reason"] = _logged_text(state.final_reason)
        self._emit_trace("run_finished", payload)
        self._log_event("run_finished", state.step_count, payload)

    def _emit_trace(self, name: str, payload: dict[str, object]) -> None:
        try:
            self.trace.emit(name, payload)
        except Exception:
            return

    def _log_event(
        self,
        event_type: str,
        step: int,
        payload: dict[str, object],
    ) -> None:
        try:
            self.logger.log(event_type, step, payload)
        except Exception:
            return


def _action_fingerprint(action: ToolCallAction) -> str:
    try:
        arguments = json.dumps(
            _json_log_safe(action.arguments),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        arguments = repr(action.arguments).encode("utf-8", errors="replace")
    return f"{action.tool_name}:{hashlib.sha256(arguments).hexdigest()}"


def _logged_arguments(arguments: object) -> object:
    safe_arguments = _json_log_safe(arguments)
    try:
        encoded = json.dumps(
            safe_arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(arguments).encode("utf-8", errors="replace")

    if len(encoded) <= MAX_LOGGED_ARGUMENT_BYTES:
        return safe_arguments
    preview = encoded.decode("utf-8", errors="ignore")[
        :MAX_LOGGED_ARGUMENT_PREVIEW_CHARS
    ]
    return {
        "truncated": True,
        "bytes": len(encoded),
        "preview": f"{preview}...",
    }


def _logged_text(value: object) -> object:
    text = str(value)
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_LOGGED_TEXT_BYTES:
        return text
    preview = encoded.decode("utf-8", errors="ignore")[:MAX_LOGGED_TEXT_PREVIEW_CHARS]
    return {
        "truncated": True,
        "bytes": len(encoded),
        "preview": f"{preview}...",
    }


def _logged_value(value: object) -> object:
    safe_value = _json_log_safe(value)
    try:
        encoded = json.dumps(
            safe_value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(value).encode("utf-8", errors="replace")

    if len(encoded) <= MAX_LOGGED_TEXT_BYTES:
        return safe_value
    preview = encoded.decode("utf-8", errors="ignore")[:MAX_LOGGED_TEXT_PREVIEW_CHARS]
    return {
        "truncated": True,
        "bytes": len(encoded),
        "preview": f"{preview}...",
    }


def _json_log_safe(
    value: object,
    seen: set[int] | None = None,
    depth: int = 0,
) -> object:
    if depth > MAX_LOGGED_VALUE_DEPTH:
        return MAX_DEPTH_EXCEEDED
    if value is None or isinstance(value, str | int | float | bool):
        return value

    seen = seen or set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return {
                str(key): _json_log_safe(item, seen, depth + 1)
                for key, item in value.items()
            }
        finally:
            seen.remove(marker)

    if isinstance(value, list | tuple):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return [_json_log_safe(item, seen, depth + 1) for item in value]
        finally:
            seen.remove(marker)

    return str(value)


def _validate_tool_action_metadata(action: ToolCallAction) -> str | None:
    if not isinstance(action.tool_name, str) or not action.tool_name.strip():
        return "tool_name must be a non-empty string"
    if len(action.tool_name.encode("utf-8")) > MAX_TOOL_ACTION_METADATA_BYTES:
        return f"tool_name exceeds {MAX_TOOL_ACTION_METADATA_BYTES} bytes"
    if not isinstance(action.call_id, str) or not action.call_id.strip():
        return "call_id must be a non-empty string"
    if len(action.call_id.encode("utf-8")) > MAX_TOOL_ACTION_METADATA_BYTES:
        return f"call_id exceeds {MAX_TOOL_ACTION_METADATA_BYTES} bytes"
    return None


def _duplicate_batch_call_id(tool_calls: list[object]) -> str | None:
    seen: set[str] = set()
    for tool_call in tool_calls:
        if not isinstance(tool_call, ToolCallAction):
            continue
        call_id = tool_call.call_id
        if not isinstance(call_id, str) or not call_id.strip():
            continue
        if call_id in seen:
            return call_id
        seen.add(call_id)
    return None


def _validate_final_action_metadata(action: FinalAction) -> str | None:
    if not isinstance(action.text, str):
        return "text must be a string"
    if not isinstance(action.status, str) or not action.status.strip():
        return "status must be a non-empty string"
    return None


def _validate_tool_result_metadata(result: ToolResult) -> list[str]:
    errors = []
    if not isinstance(result.success, bool):
        errors.append("success must be a boolean")
    if not isinstance(result.payload, Mapping):
        errors.append("payload must be a mapping")
    if not isinstance(result.summary, str):
        errors.append("summary must be a string")
    if result.error is not None and not isinstance(result.error, str):
        errors.append("error must be a string or null")
    return errors


def _conversation_message(
    role: object,
    content: object,
    metadata: object,
) -> ConversationMessage:
    return ConversationMessage(
        role=str(role),
        content=str(content),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _trace_context(trace_context: object | None) -> dict[str, object]:
    if not isinstance(trace_context, Mapping):
        return {}
    return {str(key): _logged_value(value) for key, value in trace_context.items()}


def _validate_tool_arguments(
    schema: dict[str, object],
    arguments: object,
) -> list[str]:
    if not isinstance(arguments, dict):
        return ["Tool arguments must be an object"]
    if not all(isinstance(name, str) for name in arguments):
        return ["Tool argument names must be strings"]
    if _contains_recursive_json_value(arguments):
        return ["Tool arguments cannot contain recursive values"]

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
        errors.extend(_validate_schema_value(name, value, property_schema))

    return errors


def _contains_recursive_json_value(
    value: object,
) -> bool:
    active: set[int] = set()
    stack: list[tuple[object, bool]] = [(value, False)]
    while stack:
        item, exiting = stack.pop()
        if not isinstance(item, Mapping | list | tuple):
            continue
        marker = id(item)
        if exiting:
            active.discard(marker)
            continue
        if marker in active:
            return True
        active.add(marker)
        stack.append((item, True))
        if isinstance(item, Mapping):
            stack.extend((child, False) for child in item.values())
        else:
            stack.extend((child, False) for child in item)
    return False


def _normalize_json_types(raw_type: object) -> list[str]:
    if isinstance(raw_type, str):
        return [raw_type]
    if isinstance(raw_type, list):
        return [item for item in raw_type if isinstance(item, str)]
    return []


def _format_json_types(expected_types: list[str]) -> str:
    return " or ".join(expected_types)


def _format_enum_values(values: list[object]) -> str:
    return ", ".join(str(value) for value in values)


def _validate_schema_value(
    name: str,
    value: object,
    value_schema: dict[str, object],
) -> list[str]:
    expected_types = _normalize_json_types(value_schema.get("type"))
    if expected_types and not any(
        _matches_json_type(value, item) for item in expected_types
    ):
        return [f"Argument {name} must be {_format_json_types(expected_types)}"]

    errors: list[str] = []
    enum_values = value_schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(
            f"Argument {name} must be one of: {_format_enum_values(enum_values)}"
        )
    errors.extend(_validate_array_items(name, value, value_schema))
    errors.extend(_validate_nested_object(name, value, value_schema))
    return errors


def _validate_array_items(
    name: str,
    value: object,
    property_schema: dict[str, object],
) -> list[str]:
    if not isinstance(value, list):
        return []
    items_schema = property_schema.get("items")
    if not isinstance(items_schema, dict):
        return []

    errors: list[str] = []
    for index, item in enumerate(value):
        errors.extend(_validate_schema_value(f"{name}[{index}]", item, items_schema))
    return errors


def _validate_nested_object(
    name: str,
    value: object,
    object_schema: dict[str, object],
) -> list[str]:
    if not isinstance(value, dict):
        return []

    properties = object_schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    errors: list[str] = []
    required = object_schema.get("required", [])
    if isinstance(required, list):
        for child_name in required:
            if isinstance(child_name, str) and child_name not in value:
                errors.append(f"Missing required argument: {name}.{child_name}")

    if object_schema.get("additionalProperties") is False:
        for child_name in value:
            if child_name not in properties:
                errors.append(f"Unexpected argument: {name}.{child_name}")

    for child_name, child_value in value.items():
        child_schema = properties.get(child_name)
        if not isinstance(child_schema, dict):
            continue
        errors.extend(
            _validate_schema_value(
                f"{name}.{child_name}",
                child_value,
                child_schema,
            )
        )
    return errors


def _matches_json_type(value: object, expected_type: str) -> bool:
    if expected_type == "null":
        return value is None
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
