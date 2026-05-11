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
        self.trace.emit(
            "llm_request",
            {
                "model": request["model"],
                "instructions_preview": preview(request["instructions"]),
                "input_preview": preview(request["input"]),
                "tool_names": [tool.name for tool in registry.all()],
            },
        )
        response = self.client.responses.create(**request)

        output = _read_attr(response, "output", []) or []
        tool_calls: list[ToolCallAction] = []
        for item in output:
            if _read_attr(item, "type") == "function_call":
                tool_calls.append(
                    ToolCallAction(
                        tool_name=str(_read_attr(item, "name")),
                        arguments=self._parse_arguments(
                            _read_attr(item, "arguments", "{}")
                        ),
                        call_id=str(_read_attr(item, "call_id")),
                        provider_metadata={},
                    )
                )
        if output:
            if not state.response_context_items:
                state.response_context_items = self._conversation_items(state)
            state.response_context_items.extend(
                self._serialize_output_item(item) for item in output
            )
        self.trace.emit(
            "llm_response",
            {
                "function_calls": [
                    {"name": call.tool_name, "call_id": call.call_id}
                    for call in tool_calls
                ],
                "final_preview": preview(_read_attr(response, "output_text", "")),
            },
        )
        if len(tool_calls) == 1:
            return tool_calls[0]
        if len(tool_calls) > 1:
            return ToolCallBatchAction(tool_calls=tool_calls)

        output_text = _read_attr(response, "output_text", "")
        if output_text:
            return FinalAction(text=str(output_text))

        raise LLMResponseError("Response did not contain final text or a function call")

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
                        "output": json.dumps(observation.result.model_dump()),
                    }
                    for observation in unsent
                )
                state.sent_observation_count = len(state.observations)
            return state.response_context_items

        conversation = getattr(state, "conversation", [])
        if conversation:
            return self._conversation_items(state)
        return state.goal

    def _conversation_items(self, state: RunState) -> list[dict[str, str]]:
        conversation = state.conversation[-self.config.session_context_turns :]
        return [
            {"role": turn.role, "content": turn.content}
            for turn in [*conversation, _current_user_turn(state.goal)]
        ]

    def _serialize_output_item(self, item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)

        model_dump = getattr(item, "model_dump", None)
        if model_dump is not None:
            return model_dump(exclude_none=True)

        item_type = _read_attr(item, "type")
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
            return serialized

        serialized = {"type": item_type}
        for field in ("id", "role", "content", "status"):
            value = _read_attr(item, field)
            if value is not None:
                serialized[field] = value
        return serialized

    def _parse_arguments(self, raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        parsed = json.loads(raw_arguments)
        if not isinstance(parsed, dict):
            raise LLMResponseError("Function call arguments must decode to an object")
        return parsed


def _current_user_turn(goal: str) -> ConversationMessage:
    return ConversationMessage(role="user", content=goal)
