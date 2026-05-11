import json
from types import SimpleNamespace

import pytest

from agentskeleton.config import RunConfig
from agentskeleton.core.actions import FinalAction, ToolCallAction, ToolCallBatchAction
from agentskeleton.core.llm import (
    LLMClient,
    LLMResponseError,
    MissingAPIKeyError,
    normalize_base_url,
    resolve_openai_settings,
)
from agentskeleton.core.state import ConversationMessage, RunState, ToolObservation
from agentskeleton.tools.base import Tool, ToolResult
from agentskeleton.tools.registry import ToolRegistry


class DummyTool(Tool):
    name = "read_file"
    description = "Read a file."
    risk = "read"
    args_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(self, args, context):  # pragma: no cover - not used by these tests
        raise NotImplementedError


class FakeResponses:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response) -> None:
        self.responses = FakeResponses(response)


class FakeResponsesSequence:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeSequenceClient:
    def __init__(self, responses) -> None:
        self.responses = FakeResponsesSequence(responses)


class FailingTrace:
    def emit(self, name: str, payload: dict[str, object]) -> None:
        raise OSError("trace sink unavailable")


def test_llm_client_requires_api_key_without_injected_client(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError):
        LLMClient(RunConfig())


def test_normalize_base_url_accepts_full_responses_endpoint() -> None:
    endpoint = (
        "https://my-ai-resource.services.ai.azure.com/api/projects/demo/"
        "openai/v1/responses"
    )

    assert normalize_base_url(endpoint) == (
        "https://my-ai-resource.services.ai.azure.com/api/projects/demo/"
        "openai/v1/"
    )


def test_normalize_base_url_treats_blank_as_unset() -> None:
    assert normalize_base_url("   ") is None


def test_resolve_openai_settings_uses_azure_environment(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    monkeypatch.setenv(
        "AZURE_EXISTING_AIPROJECT_ENDPOINT",
        "https://example.openai.azure.com/openai/v1/",
    )

    settings = resolve_openai_settings(RunConfig())

    assert settings.api_key == "dummy-key"
    assert settings.base_url == "https://example.openai.azure.com/openai/v1/"


def test_llm_client_sends_tool_schemas_and_parses_final_text(tmp_path) -> None:
    response = SimpleNamespace(id="resp-1", output_text="done", output=[])
    fake_client = FakeClient(response)
    state = RunState(run_id="run-1", workspace=tmp_path, goal="finish")

    action = LLMClient(RunConfig(workspace=tmp_path), client=fake_client).next_action(
        state,
        ToolRegistry([DummyTool()]),
    )

    assert isinstance(action, FinalAction)
    assert action.text == "done"
    call = fake_client.responses.calls[0]
    assert call["model"] == "gpt-5.5"
    assert "registered function tools" in call["instructions"]
    assert "previous_response_id" not in call
    assert call["input"] == "finish"
    assert call["tools"][0]["name"] == "read_file"
    assert call["reasoning"] == {"effort": "low"}
    assert call["text"] == {"verbosity": "low"}


def test_llm_client_continues_when_trace_sink_fails(tmp_path) -> None:
    response = SimpleNamespace(id="resp-1", output_text="done", output=[])
    fake_client = FakeClient(response)
    state = RunState(run_id="run-1", workspace=tmp_path, goal="finish")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=fake_client,
        trace=FailingTrace(),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert action.text == "done"
    assert len(fake_client.responses.calls) == 1


def test_llm_client_parses_function_call(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments='{"path": "README.md"}',
                call_id="call-1",
            )
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, ToolCallAction)
    assert action.tool_name == "read_file"
    assert action.arguments == {"path": "README.md"}
    assert action.call_id == "call-1"


def test_llm_client_rejects_function_call_without_name(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                arguments='{"path": "README.md"}',
                call_id="call-1",
            )
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(LLMResponseError, match="Function call missing name"):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))


def test_llm_client_rejects_function_call_without_call_id(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments='{"path": "README.md"}',
            )
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(LLMResponseError, match="Function call missing call_id"):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))


def test_llm_client_rejects_invalid_function_call_arguments_json(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments="{bad json",
                call_id="call-1",
            )
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(
        LLMResponseError,
        match="Function call arguments must be valid JSON",
    ):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))


def test_llm_client_parses_multiple_function_calls_as_batch(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments='{"path": "README.md"}',
                call_id="call-1",
            ),
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments='{"path": "AGENTS.md"}',
                call_id="call-2",
            ),
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, ToolCallBatchAction)
    assert [tool_call.call_id for tool_call in action.tool_calls] == [
        "call-1",
        "call-2",
    ]


def test_llm_client_sends_tool_observations_as_stateless_input(tmp_path) -> None:
    first_response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments='{"path": "README.md"}',
                call_id="call-1",
            )
        ],
    )
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeSequenceClient([first_response, response])
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="read",
    )
    client = LLMClient(RunConfig(workspace=tmp_path), client=fake_client)

    client.next_action(state, ToolRegistry([DummyTool()]))
    state.observations.append(
        ToolObservation(
            call_id="call-1",
            tool_name="read_file",
            policy_decision="allow",
            result=ToolResult(
                success=True,
                payload={"content": "hello"},
                summary="ok",
            ),
        )
    )
    client.next_action(state, ToolRegistry([DummyTool()]))

    call = fake_client.responses.calls[1]
    assert "previous_response_id" not in call
    assert call["input"] == [
        {"role": "user", "content": "read"},
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"path": "README.md"}',
            "call_id": "call-1",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": (
                '{"success": true, "payload": {"content": "hello"}, '
                '"summary": "ok", "error": null}'
            ),
        }
    ]


def test_llm_client_skips_output_items_without_context_type(tmp_path) -> None:
    first_response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(content="provider note"),
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments='{"path": "README.md"}',
                call_id="call-1",
            ),
        ],
    )
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeSequenceClient([first_response, response])
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")
    client = LLMClient(RunConfig(workspace=tmp_path), client=fake_client)

    client.next_action(state, ToolRegistry([DummyTool()]))
    state.observations.append(
        ToolObservation(
            call_id="call-1",
            tool_name="read_file",
            policy_decision="allow",
            result=ToolResult(success=True, payload={}, summary="ok"),
        )
    )
    client.next_action(state, ToolRegistry([DummyTool()]))

    assert fake_client.responses.calls[1]["input"] == [
        {"role": "user", "content": "read"},
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"path": "README.md"}',
            "call_id": "call-1",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": (
                '{"success": true, "payload": {}, '
                '"summary": "ok", "error": null}'
            ),
        },
    ]


def test_llm_client_serializes_non_json_tool_observation_payloads(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeClient(response)
    marker = object()
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="read",
        observations=[
            ToolObservation(
                call_id="call-1",
                tool_name="read_file",
                policy_decision="allow",
                result=ToolResult(
                    success=True,
                    payload={"workspace": tmp_path, "raw": marker},
                    summary="ok",
                ),
            )
        ],
    )

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=fake_client,
    ).next_action(state, ToolRegistry([DummyTool()]))

    call = fake_client.responses.calls[0]
    observation_output = json.loads(call["input"][1]["output"])
    assert isinstance(action, FinalAction)
    assert observation_output["payload"]["workspace"] == str(tmp_path)
    assert observation_output["payload"]["raw"] == str(marker)


def test_llm_client_sends_transcript_context(
    tmp_path,
) -> None:
    response = SimpleNamespace(id="resp-2", output_text="continued", output=[])
    fake_client = FakeClient(response)
    state = RunState(
        run_id="run-2",
        workspace=tmp_path,
        goal="what next?",
        conversation=[
            ConversationMessage(role="user", content="remember alpha"),
            ConversationMessage(role="assistant", content="alpha stored"),
        ],
    )

    LLMClient(RunConfig(workspace=tmp_path), client=fake_client).next_action(
        state,
        ToolRegistry([DummyTool()]),
    )

    call = fake_client.responses.calls[0]
    assert "previous_response_id" not in call
    assert call["input"] == [
        {"role": "user", "content": "remember alpha"},
        {"role": "assistant", "content": "alpha stored"},
        {"role": "user", "content": "what next?"},
    ]


def test_llm_client_limits_transcript_context_to_recent_turns(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="continued", output=[])
    fake_client = FakeClient(response)
    state = RunState(
        run_id="run-2",
        workspace=tmp_path,
        goal="current request",
        conversation=[
            ConversationMessage(role="user", content="old user"),
            ConversationMessage(role="assistant", content="old answer"),
            ConversationMessage(role="user", content="recent user"),
            ConversationMessage(role="assistant", content="recent answer"),
        ],
    )

    LLMClient(
        RunConfig(workspace=tmp_path, session_context_turns=2),
        client=fake_client,
    ).next_action(
        state,
        ToolRegistry([DummyTool()]),
    )

    call = fake_client.responses.calls[0]
    assert call["input"] == [
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "current request"},
    ]


def test_llm_client_keeps_sticky_summary_when_limiting_transcript_context(
    tmp_path,
) -> None:
    response = SimpleNamespace(id="resp-2", output_text="continued", output=[])
    fake_client = FakeClient(response)
    state = RunState(
        run_id="run-2",
        workspace=tmp_path,
        goal="current request",
        conversation=[
            ConversationMessage(
                role="user",
                content="Prior conversation summary:\n- user: old user",
                metadata={"sticky_context": True},
            ),
            ConversationMessage(role="user", content="old user"),
            ConversationMessage(role="assistant", content="old answer"),
            ConversationMessage(role="user", content="recent user"),
            ConversationMessage(role="assistant", content="recent answer"),
        ],
    )

    LLMClient(
        RunConfig(workspace=tmp_path, session_context_turns=2),
        client=fake_client,
    ).next_action(
        state,
        ToolRegistry([DummyTool()]),
    )

    call = fake_client.responses.calls[0]
    assert call["input"] == [
        {
            "role": "user",
            "content": "Prior conversation summary:\n- user: old user",
        },
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "current request"},
    ]
