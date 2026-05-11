from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from agentskeleton.config import RunConfig, dotenv_values
from agentskeleton.core.actions import FinalAction, ToolCallAction, ToolCallBatchAction
from agentskeleton.core.state import ConversationMessage, RunState
from agentskeleton.core.trace import NullTraceSink, TraceSink, preview
from agentskeleton.tools.registry import ToolRegistry

AGENT_INSTRUCTIONS = (
    "You are a local CLI agent running inside the configured workspace.\n"
    "Use only the registered function tools in this request. When asked what "
    "tools you have, name only those registered function tools.\n"
    "Do not mention host, developer, or internal orchestration tools that are "
    "not registered in this request.\n"
    "Do not reveal hidden chain-of-thought or private reasoning. You may "
    "summarize operational steps, tool calls, observations, assumptions, and "
    "final decisions."
)
MAX_TOOL_RESULT_OUTPUT_BYTES = 1_048_576
MAX_TOOL_RESULT_OUTPUT_PREVIEW_CHARS = 512
MAX_RESPONSE_OUTPUT_ITEMS = 100
MAX_RESPONSE_CONTEXT_ITEMS = 200
MAX_RESPONSE_CONTEXT_ANCHOR_ITEMS = 50
MAX_MESSAGE_CONTENT_PARTS = 200
MAX_FUNCTION_CALL_ARGUMENT_BYTES = 2_097_152
MAX_CONVERSATION_CONTENT_CHARS = 4_096
MAX_CONVERSATION_ROLE_CHARS = 64
ALLOWED_CONVERSATION_ROLES = {"assistant", "developer", "system", "user"}
MAX_FUNCTION_CALL_METADATA_BYTES = 512
MAX_CONTEXT_METADATA_CHARS = 512
MAX_JSON_SAFE_DEPTH = 64
MAX_JSON_SAFE_ITEMS = 200
MAX_DEPTH_EXCEEDED = "<max-depth-exceeded>"
TRUNCATED_ITEMS_KEY = "__truncated_items__"


class MissingAPIKeyError(RuntimeError):
    """Raised when a live OpenAI client is requested without credentials."""


class LLMResponseError(RuntimeError):
    """Raised when a model response cannot be mapped to an internal action."""


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    base_url: str | None


def normalize_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None

    normalized = base_url.strip()
    if not normalized:
        return None
    normalized = normalized.rstrip("/")
    if normalized.endswith("/responses"):
        normalized = normalized[: -len("/responses")]
    if not normalized.endswith("/"):
        normalized = f"{normalized}/"
    return normalized


def resolve_openai_settings(config: RunConfig) -> OpenAISettings:
    dotenv = dotenv_values()
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or dotenv.get("OPENAI_API_KEY")
        or dotenv.get("AZURE_OPENAI_API_KEY")
    )
    if not api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for live LLM runs"
        )

    base_url = (
        config.base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_EXISTING_AIPROJECT_ENDPOINT")
        or dotenv.get("OPENAI_BASE_URL")
        or dotenv.get("AZURE_OPENAI_ENDPOINT")
        or dotenv.get("AZURE_EXISTING_AIPROJECT_ENDPOINT")
    )
    return OpenAISettings(api_key=api_key, base_url=normalize_base_url(base_url))


def _read_attr(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _response_output_items(response: Any) -> list[Any]:
    output = _read_attr(response, "output", []) or []
    if isinstance(output, dict):
        return [output]
    if isinstance(output, list | tuple):
        items = list(output)
        if len(items) > MAX_RESPONSE_OUTPUT_ITEMS:
            raise LLMResponseError(
                "Response output contains too many items "
                f"({len(items)} > {MAX_RESPONSE_OUTPUT_ITEMS})"
            )
        return items
    raise LLMResponseError("Response output must be a list of items")


def _response_final_text(response: Any, output: list[Any]) -> str:
    output_text = _read_attr(response, "output_text", "")
    if output_text:
        return _bounded_final_text(str(output_text))

    parts: list[str] = []
    for item in output:
        if _read_attr(item, "type") != "message":
            continue
        parts.extend(_message_content_text_parts(_read_attr(item, "content")))
    return _bounded_final_text("\n".join(part for part in parts if part))


def _message_content_text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list | tuple):
        item_limit = _message_content_part_limit(len(content))
        text_parts = [
            text
            for part in content[:item_limit]
            if (text := _message_content_part_text(part)) is not None
        ]
        omitted = len(content) - item_limit
        if omitted > 0:
            text_parts.append(f"[truncated {omitted} content parts]")
        return text_parts
    text = _message_content_part_text(content)
    return [] if text is None else [text]


def _message_content_part_text(part: Any) -> str | None:
    if isinstance(part, str):
        return part
    part_type = _read_attr(part, "type")
    if part_type == "refusal":
        refusal = _read_attr(part, "refusal")
        return refusal if isinstance(refusal, str) else None
    text = _read_attr(part, "text")
    if not isinstance(text, str):
        return None
    if part_type in (None, "output_text", "text"):
        return text
    return None


def _message_content_part_limit(total_items: int) -> int:
    if total_items <= MAX_MESSAGE_CONTENT_PARTS:
        return total_items
    return MAX_MESSAGE_CONTENT_PARTS - 1


class LLMClient:
    def __init__(
        self,
        config: RunConfig,
        client: Any | None = None,
        trace: TraceSink | None = None,
    ) -> None:
        self.config = config
        self.trace = trace or NullTraceSink()
        if client is not None:
            self.client = client
            return

        settings = resolve_openai_settings(config)

        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": settings.api_key}
        if settings.base_url is not None:
            kwargs["base_url"] = settings.base_url
        self.client = OpenAI(**kwargs)

    def next_action(self, state: RunState, registry: ToolRegistry):
        request = self._build_request(state, registry)
        self._emit_trace(
            "llm_request",
            {
                "model": request["model"],
                "instructions_preview": preview(request["instructions"]),
                "input_preview": preview(request["input"]),
                "tool_names": [tool.name for tool in registry.all()],
            },
        )
        response = self.client.responses.create(**request)

        output = _response_output_items(response)
        tool_calls: list[ToolCallAction] = []
        for item in output:
            if _read_attr(item, "type") == "function_call":
                tool_calls.append(self._parse_function_call(item))
        final_text = _response_final_text(response, output)
        if output:
            if not state.response_context_items:
                state.response_context_items = self._conversation_items(state)
            state.response_context_items.extend(
                serialized
                for item in output
                if (serialized := self._serialize_output_item(item)) is not None
            )
            state.response_context_items = _trim_response_context_items(
                state.response_context_items
            )
        self._emit_trace(
            "llm_response",
            {
                "function_calls": [
                    {"name": call.tool_name, "call_id": call.call_id}
                    for call in tool_calls
                ],
                "final_preview": preview(final_text),
            },
        )
        if len(tool_calls) == 1:
            return tool_calls[0]
        if len(tool_calls) > 1:
            return ToolCallBatchAction(tool_calls=tool_calls)

        if final_text:
            return FinalAction(text=final_text)

        raise LLMResponseError("Response did not contain final text or a function call")

    def _emit_trace(self, name: str, payload: dict[str, Any]) -> None:
        try:
            self.trace.emit(name, payload)
        except Exception:
            return

    def _build_request(self, state: RunState, registry: ToolRegistry) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.model,
            "instructions": AGENT_INSTRUCTIONS,
            "input": self._build_input(state),
            "tools": registry.to_openai_tools(),
            "reasoning": {"effort": self.config.reasoning_effort},
            "text": {"verbosity": self.config.text_verbosity},
        }
        return request

    def _build_input(self, state: RunState) -> str | list[dict[str, Any]]:
        unsent = state.observations[state.sent_observation_count :]
        if state.response_context_items or unsent:
            if not state.response_context_items:
                state.response_context_items = self._conversation_items(state)
            if unsent:
                state.response_context_items.extend(
                    {
                        "type": "function_call_output",
                        "call_id": observation.call_id,
                        "output": _serialize_tool_result(observation.result),
                    }
                    for observation in unsent
                )
                state.sent_observation_count = len(state.observations)
            state.response_context_items = _trim_response_context_items(
                state.response_context_items
            )
            return state.response_context_items

        conversation = getattr(state, "conversation", [])
        if conversation:
            return _trim_response_context_items(self._conversation_items(state))
        return state.goal

    def _conversation_items(self, state: RunState) -> list[dict[str, str]]:
        sticky_candidates = [
            turn
            for turn in state.conversation
            if turn.metadata.get("sticky_context") is True
        ]
        sticky_limit = min(
            self.config.session_context_turns,
            MAX_RESPONSE_CONTEXT_ANCHOR_ITEMS,
        )
        sticky_context = (
            sticky_candidates[-sticky_limit:] if sticky_limit > 0 else []
        )
        conversation_budget = max(
            MAX_RESPONSE_CONTEXT_ITEMS - len(sticky_context) - 1,
            0,
        )
        conversation_limit = min(
            self.config.session_context_turns,
            conversation_budget,
        )
        conversation_candidates = [
            turn
            for turn in state.conversation
            if turn.metadata.get("sticky_context") is not True
        ]
        conversation = (
            conversation_candidates[-conversation_limit:]
            if conversation_limit > 0
            else []
        )
        return [
            {
                "role": _safe_conversation_role(turn.role),
                "content": _bounded_conversation_content(turn.content),
            }
            for turn in [*sticky_context, *conversation, _current_user_turn(state.goal)]
        ]

    def _serialize_output_item(self, item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict):
            return _normalize_context_item(dict(item))

        model_dump = getattr(item, "model_dump", None)
        if model_dump is not None:
            return _normalize_context_item(model_dump(exclude_none=True))

        item_type = _read_attr(item, "type")
        if not isinstance(item_type, str) or not item_type.strip():
            return None
        if item_type == "function_call":
            serialized = {
                "type": "function_call",
                "name": _read_attr(item, "name"),
                "arguments": _read_attr(item, "arguments"),
                "call_id": _read_attr(item, "call_id"),
            }
            item_id = _read_attr(item, "id")
            if item_id is not None:
                serialized["id"] = item_id
            status = _read_attr(item, "status")
            if status is not None:
                serialized["status"] = status
            return _normalize_context_item(serialized)

        serialized = {"type": item_type}
        for field in ("id", "role", "content", "status"):
            value = _read_attr(item, field)
            if value is not None:
                serialized[field] = value
        return _normalize_context_item(serialized)

    def _parse_function_call(self, item: Any) -> ToolCallAction:
        name = _read_attr(item, "name")
        if not isinstance(name, str) or not name.strip():
            raise LLMResponseError("Function call missing name")
        if len(name.encode("utf-8")) > MAX_FUNCTION_CALL_METADATA_BYTES:
            raise LLMResponseError(
                f"Function call name exceeds {MAX_FUNCTION_CALL_METADATA_BYTES} bytes"
            )

        call_id = _read_attr(item, "call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise LLMResponseError("Function call missing call_id")
        if len(call_id.encode("utf-8")) > MAX_FUNCTION_CALL_METADATA_BYTES:
            raise LLMResponseError(
                "Function call call_id exceeds "
                f"{MAX_FUNCTION_CALL_METADATA_BYTES} bytes"
            )

        return ToolCallAction(
            tool_name=name,
            arguments=self._parse_arguments(_read_attr(item, "arguments", "{}")),
            call_id=call_id,
            provider_metadata={},
        )

    def _parse_arguments(self, raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            try:
                if _json_exceeds_depth(raw_arguments, MAX_JSON_SAFE_DEPTH):
                    raise LLMResponseError(
                        "Function call arguments are too deeply nested"
                    )
                if _json_size(raw_arguments) > MAX_FUNCTION_CALL_ARGUMENT_BYTES:
                    raise LLMResponseError(
                        "Function call arguments are too large "
                        f"(max {MAX_FUNCTION_CALL_ARGUMENT_BYTES} bytes)"
                    )
            except LLMResponseError:
                raise
            except Exception as exc:
                raise LLMResponseError(
                    "Function call arguments could not be inspected"
                ) from exc
            return raw_arguments
        if isinstance(raw_arguments, str):
            if len(raw_arguments.encode("utf-8")) > MAX_FUNCTION_CALL_ARGUMENT_BYTES:
                raise LLMResponseError(
                    "Function call arguments are too large "
                    f"(max {MAX_FUNCTION_CALL_ARGUMENT_BYTES} bytes)"
                )
        try:
            parsed = json.loads(raw_arguments)
        except RecursionError as exc:
            raise LLMResponseError(
                "Function call arguments are too deeply nested"
            ) from exc
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(
                "Function call arguments must be valid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise LLMResponseError("Function call arguments must decode to an object")
        if _json_exceeds_depth(parsed, MAX_JSON_SAFE_DEPTH):
            raise LLMResponseError("Function call arguments are too deeply nested")
        return parsed


def _current_user_turn(goal: str) -> ConversationMessage:
    return ConversationMessage(role="user", content=goal)


def _bounded_conversation_content(content: str) -> str:
    if len(content) <= MAX_CONVERSATION_CONTENT_CHARS:
        return content
    omitted = len(content) - MAX_CONVERSATION_CONTENT_CHARS
    return (
        f"{content[:MAX_CONVERSATION_CONTENT_CHARS]}"
        f"\n[truncated {omitted} characters]"
    )


def _bounded_final_text(text: str) -> str:
    return _bounded_conversation_content(text)


def _safe_conversation_role(role: object) -> str:
    if not isinstance(role, str):
        return "user"
    normalized = role.strip().lower()
    if len(normalized) > MAX_CONVERSATION_ROLE_CHARS:
        return "user"
    if normalized not in ALLOWED_CONVERSATION_ROLES:
        return "user"
    return normalized


def _trim_response_context_items(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(items) <= MAX_RESPONSE_CONTEXT_ITEMS:
        return items
    anchors = _leading_context_anchors(items)
    tail_budget = MAX_RESPONSE_CONTEXT_ITEMS - len(anchors)
    if tail_budget <= 0:
        return anchors[:MAX_RESPONSE_CONTEXT_ITEMS]
    return [*anchors, *items[-tail_budget:]]


def _leading_context_anchors(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for item in items:
        if len(anchors) >= MAX_RESPONSE_CONTEXT_ANCHOR_ITEMS:
            break
        if not _is_conversation_context_item(item):
            break
        anchors.append(item)
    return anchors


def _is_conversation_context_item(item: dict[str, Any]) -> bool:
    return (
        "type" not in item
        and isinstance(item.get("role"), str)
        and "content" in item
    )


def _is_context_item(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = item.get("type")
    role = item.get("role")
    return (
        isinstance(item_type, str)
        and bool(item_type.strip())
        or isinstance(role, str)
        and bool(role.strip())
    )


def _normalize_context_item(item: dict[str, Any]) -> dict[str, Any] | None:
    if not _is_context_item(item):
        return None
    if item.get("type") == "function_call" and isinstance(item.get("arguments"), dict):
        item["arguments"] = json.dumps(_json_safe(item["arguments"]))
    for field in ("id", "name", "call_id", "status", "type"):
        if field in item:
            item[field] = _bounded_context_metadata(item[field])
    if "role" in item:
        item["role"] = _safe_conversation_role(item["role"])
    if "content" in item:
        item["content"] = _bounded_context_content(item["content"])
    return _json_safe(item)


def _bounded_context_metadata(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= MAX_CONTEXT_METADATA_CHARS:
        return value
    omitted = len(value) - MAX_CONTEXT_METADATA_CHARS
    return f"{value[:MAX_CONTEXT_METADATA_CHARS]}\n[truncated {omitted} characters]"


def _bounded_context_content(
    value: Any,
    seen: set[int] | None = None,
    depth: int = 0,
) -> Any:
    if depth > MAX_JSON_SAFE_DEPTH:
        return MAX_DEPTH_EXCEEDED
    if isinstance(value, str):
        return _bounded_conversation_content(value)
    seen = seen or set()
    if isinstance(value, list):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            item_limit = _json_safe_item_limit(len(value))
            safe_items = [
                _bounded_context_content(item, seen, depth + 1)
                for item in value[:item_limit]
            ]
            omitted = len(value) - item_limit
            if omitted > 0:
                safe_items.append(_truncated_items_marker(len(value), omitted))
            return safe_items
        finally:
            seen.remove(marker)
    if isinstance(value, tuple):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            item_limit = _json_safe_item_limit(len(value))
            safe_items = [
                _bounded_context_content(item, seen, depth + 1)
                for item in value[:item_limit]
            ]
            omitted = len(value) - item_limit
            if omitted > 0:
                safe_items.append(_truncated_items_marker(len(value), omitted))
            return safe_items
        finally:
            seen.remove(marker)
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            safe_items: dict[Any, Any] = {}
            item_limit = _json_safe_item_limit(len(value))
            for index, (key, item) in enumerate(value.items()):
                if index >= item_limit:
                    continue
                safe_items[key] = _bounded_context_content(item, seen, depth + 1)
            omitted = len(value) - item_limit
            if omitted:
                safe_items[TRUNCATED_ITEMS_KEY] = _truncated_items_marker(
                    len(value),
                    omitted,
                )
            return safe_items
        finally:
            seen.remove(marker)
    return value


def _serialize_tool_result(result: Any) -> str:
    payload = {
        "success": result.success,
        "payload": result.payload,
        "summary": result.summary,
        "error": result.error,
    }
    serialized = json.dumps(_json_safe(payload))
    serialized_bytes = serialized.encode("utf-8")
    if len(serialized_bytes) <= MAX_TOOL_RESULT_OUTPUT_BYTES:
        return serialized

    return json.dumps(
        {
            "success": result.success,
            "payload": {
                "truncated": True,
                "bytes": len(serialized_bytes),
                "preview": f"{serialized[:MAX_TOOL_RESULT_OUTPUT_PREVIEW_CHARS]}...",
            },
            "summary": _bounded_tool_result_text(result.summary),
            "error": (
                _bounded_tool_result_text(result.error)
                if result.error is not None
                else None
            ),
        },
        separators=(",", ":"),
    )


def _bounded_tool_result_text(value: str) -> str:
    if len(value) <= MAX_TOOL_RESULT_OUTPUT_PREVIEW_CHARS:
        return value
    return f"{value[:MAX_TOOL_RESULT_OUTPUT_PREVIEW_CHARS]}..."


def _json_size(value: Any) -> int:
    return len(json.dumps(_json_size_safe(value), default=str).encode("utf-8"))


def _json_exceeds_depth(value: Any, max_depth: int) -> bool:
    stack: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    while stack:
        item, depth = stack.pop()
        if depth > max_depth:
            return True
        if isinstance(item, dict):
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list | tuple):
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            stack.extend((child, depth + 1) for child in item)
    return False


def _json_safe(value: Any, seen: set[int] | None = None, depth: int = 0) -> Any:
    if depth > MAX_JSON_SAFE_DEPTH:
        return MAX_DEPTH_EXCEEDED
    if value is None or isinstance(value, str | int | float | bool):
        return value

    seen = seen or set()
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            safe_items: dict[str, Any] = {}
            item_limit = _json_safe_item_limit(len(value))
            for index, (key, item) in enumerate(value.items()):
                if index >= item_limit:
                    continue
                safe_items[str(key)] = _json_safe(item, seen, depth + 1)
            omitted = len(value) - item_limit
            if omitted:
                safe_items[TRUNCATED_ITEMS_KEY] = _truncated_items_marker(
                    len(value),
                    omitted,
                )
            return safe_items
        finally:
            seen.remove(marker)

    if isinstance(value, list | tuple):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            item_limit = _json_safe_item_limit(len(value))
            safe_items = [
                _json_safe(item, seen, depth + 1) for item in value[:item_limit]
            ]
            omitted = len(value) - item_limit
            if omitted > 0:
                safe_items.append(_truncated_items_marker(len(value), omitted))
            return safe_items
        finally:
            seen.remove(marker)

    return str(value)


def _json_size_safe(value: Any, seen: set[int] | None = None, depth: int = 0) -> Any:
    if depth > MAX_JSON_SAFE_DEPTH:
        return MAX_DEPTH_EXCEEDED
    if value is None or isinstance(value, str | int | float | bool):
        return value

    seen = seen or set()
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return {
                str(key): _json_size_safe(item, seen, depth + 1)
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
            return [_json_size_safe(item, seen, depth + 1) for item in value]
        finally:
            seen.remove(marker)

    return str(value)


def _json_safe_item_limit(total_items: int) -> int:
    if total_items <= MAX_JSON_SAFE_ITEMS:
        return total_items
    return MAX_JSON_SAFE_ITEMS - 1


def _truncated_items_marker(total_items: int, omitted: int) -> dict[str, object]:
    return {
        "truncated": True,
        "items": total_items,
        "omitted": omitted,
    }
