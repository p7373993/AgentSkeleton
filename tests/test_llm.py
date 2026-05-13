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


def test_normalize_base_url_accepts_responses_endpoint_with_trailing_slash() -> None:
    endpoint = (
        " https://my-ai-resource.services.ai.azure.com/api/projects/demo/"
        "openai/v1/responses/ "
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


def test_llm_client_bounds_large_output_text(tmp_path) -> None:
    large_output = "x" * 20_000
    response = SimpleNamespace(id="resp-1", output_text=large_output, output=[])
    state = RunState(run_id="run-1", workspace=tmp_path, goal="finish")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert len(action.text) < 5_000
    assert action.text.startswith("x" * 40)
    assert "[truncated" in action.text
    assert large_output not in action.text


def test_llm_client_serializes_unstringable_output_text(tmp_path) -> None:
    class UnstringableOutputText:
        def __str__(self) -> str:
            raise RuntimeError("output text unavailable")

    response = SimpleNamespace(
        id="resp-1",
        output_text=UnstringableOutputText(),
        output=[],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="finish")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert action.text == "<uninspectable>"


def test_llm_client_parses_final_text_from_message_output_content(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "done from message content",
                    }
                ],
            }
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="finish")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert action.text == "done from message content"
    assert state.response_context_items == [
        {"role": "user", "content": "finish"},
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "done from message content",
                }
            ],
        },
    ]


def test_llm_client_bounds_final_text_message_content_parts(tmp_path) -> None:
    content = [
        {"type": "output_text", "text": f"part-{index:03d}"}
        for index in range(500)
    ]
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": content,
            }
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="finish")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert "part-000" in action.text
    assert "part-300" not in action.text
    assert "[truncated 301 content parts]" in action.text


def test_llm_client_parses_refusal_text_from_message_output_content(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "refusal",
                        "refusal": "I cannot help with that request.",
                    }
                ],
            }
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="finish")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert action.text == "I cannot help with that request."


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


def test_llm_client_accepts_single_output_item_mapping(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output={
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"path": "README.md"}',
            "call_id": "call-1",
        },
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, ToolCallAction)
    assert action.tool_name == "read_file"
    assert state.response_context_items == [
        {"role": "user", "content": "read"},
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"path": "README.md"}',
            "call_id": "call-1",
        },
    ]


def test_llm_client_serializes_mapping_function_call_arguments_for_context(
    tmp_path,
) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output={
            "type": "function_call",
            "name": "read_file",
            "arguments": {"path": "README.md"},
            "call_id": "call-1",
        },
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, ToolCallAction)
    assert action.arguments == {"path": "README.md"}
    assert state.response_context_items == [
        {"role": "user", "content": "read"},
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"path": "README.md"}',
            "call_id": "call-1",
        },
    ]


def test_llm_client_serializes_object_function_call_arguments_for_context(
    tmp_path,
) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments={"path": "README.md"},
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
    assert action.arguments == {"path": "README.md"}
    assert state.response_context_items == [
        {"role": "user", "content": "read"},
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"path": "README.md"}',
            "call_id": "call-1",
        },
    ]


def test_llm_client_rejects_invalid_output_shape(tmp_path) -> None:
    response = SimpleNamespace(id="resp-1", output_text="", output="not a list")
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(LLMResponseError, match="Response output must be a list"):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))


def test_llm_client_rejects_too_many_response_output_items(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="done",
        output=[
            {"type": "message", "role": "assistant", "content": "note"}
            for _ in range(101)
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(
        LLMResponseError,
        match="Response output contains too many items",
    ):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))

    assert state.response_context_items == []


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "a" * 513, "Function call name exceeds 512 bytes"),
        ("call_id", "a" * 513, "Function call call_id exceeds 512 bytes"),
    ],
)
def test_llm_client_rejects_oversized_function_call_metadata(
    tmp_path,
    field: str,
    value: str,
    message: str,
) -> None:
    item = {
        "type": "function_call",
        "name": "read_file",
        "arguments": "{}",
        "call_id": "call-1",
    }
    item[field] = value
    response = SimpleNamespace(id="resp-1", output_text="", output=[item])
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(LLMResponseError, match=message):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))

    assert state.response_context_items == []


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


def test_llm_client_rejects_too_deep_function_call_arguments_json(tmp_path) -> None:
    arguments = "{}"
    for _ in range(1_200):
        arguments = f'{{"child":{arguments}}}'
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments=arguments,
                call_id="call-1",
            )
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(
        LLMResponseError,
        match="Function call arguments are too deeply nested",
    ):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))


def test_llm_client_rejects_oversized_function_call_arguments(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments='{"content":"' + ("x" * 2_100_000) + '"}',
                call_id="call-1",
            )
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(
        LLMResponseError,
        match="Function call arguments are too large",
    ):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))

    assert state.response_context_items == []


def test_llm_client_rejects_oversized_mapping_function_call_arguments(
    tmp_path,
) -> None:
    large_arguments = {
        f"{index:05d}_{'x' * 300}": "value"
        for index in range(9_000)
    }
    assert len(json.dumps(large_arguments).encode("utf-8")) > 2_097_152
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments=large_arguments,
                call_id="call-1",
            )
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(
        LLMResponseError,
        match="Function call arguments are too large",
    ):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))

    assert state.response_context_items == []


def test_llm_client_rejects_uninspectable_mapping_function_call_arguments(
    tmp_path,
) -> None:
    class ExplodingValuesArguments(dict):
        def values(self):  # type: ignore[override]
            raise RuntimeError("argument values unavailable")

    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="read_file",
                arguments=ExplodingValuesArguments({"path": "README.md"}),
                call_id="call-1",
            )
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    with pytest.raises(
        LLMResponseError,
        match="Function call arguments could not be inspected",
    ):
        LLMClient(
            RunConfig(workspace=tmp_path),
            client=FakeClient(response),
        ).next_action(state, ToolRegistry([DummyTool()]))

    assert state.response_context_items == []


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


def test_llm_client_bounds_accumulated_response_context_items(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeClient(response)
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="read",
        response_context_items=[
            {"role": "user", "content": "read"},
            *[
                {"type": "message", "role": "assistant", "content": f"old-{index}"}
                for index in range(250)
            ],
        ],
        observations=[
            ToolObservation(
                call_id="call-250",
                tool_name="read_file",
                policy_decision="allow",
                result=ToolResult(
                    success=True,
                    payload={"content": "latest"},
                    summary="ok",
                ),
            )
        ],
    )

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=fake_client,
    ).next_action(state, ToolRegistry([DummyTool()]))

    request_input = fake_client.responses.calls[0]["input"]
    assert isinstance(action, FinalAction)
    assert len(request_input) == 200
    assert len(state.response_context_items) == 200
    assert request_input[0] == {"role": "user", "content": "read"}
    assert "old-0" not in json.dumps(request_input)
    assert "old-249" in json.dumps(request_input)
    assert request_input[-1]["type"] == "function_call_output"
    assert request_input[-1]["call_id"] == "call-250"
    assert '"content": "latest"' in request_input[-1]["output"]


def test_llm_client_bounds_response_context_after_output_items(tmp_path) -> None:
    response = SimpleNamespace(
        id="resp-2",
        output_text="done",
        output=[
            {"type": "message", "role": "assistant", "content": "fresh-1"},
            {"type": "message", "role": "assistant", "content": "fresh-2"},
        ],
    )
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="read",
        response_context_items=[
            {"role": "user", "content": "read"},
            *[
                {"type": "message", "role": "assistant", "content": f"old-{index}"}
                for index in range(198)
            ],
        ],
    )

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    serialized_context = json.dumps(state.response_context_items)
    assert isinstance(action, FinalAction)
    assert len(state.response_context_items) == 200
    assert state.response_context_items[0] == {"role": "user", "content": "read"}
    assert "old-0" not in serialized_context
    assert "fresh-1" in serialized_context
    assert "fresh-2" in serialized_context


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


def test_llm_client_serializes_non_json_context_item_values(tmp_path) -> None:
    marker = object()
    response = SimpleNamespace(
        id="resp-1",
        output_text="done",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": marker,
            }
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert state.response_context_items == [
        {"role": "user", "content": "read"},
        {
            "type": "message",
            "role": "assistant",
            "content": str(marker),
        },
    ]


def test_llm_client_bounds_large_context_item_content(tmp_path) -> None:
    large_content = "x" * 10_000
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": large_content,
            },
            {
                "type": "function_call",
                "name": "read_file",
                "arguments": '{"path": "README.md"}',
                "call_id": "call-1",
            },
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    stored_content = state.response_context_items[1]["content"]
    assert isinstance(action, ToolCallAction)
    assert isinstance(stored_content, str)
    assert len(stored_content) < 4_200
    assert stored_content.startswith("x" * 40)
    assert "[truncated 5904 characters]" in stored_content
    assert large_content not in str(state.response_context_items)


def test_llm_client_bounds_large_context_item_metadata(tmp_path) -> None:
    large_id = "i" * 10_000
    large_status = "s" * 10_000
    response = SimpleNamespace(
        id="resp-1",
        output_text="done",
        output=[
            {
                "type": "message",
                "id": large_id,
                "role": "assistant",
                "status": large_status,
                "content": "done",
            }
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    stored_item = state.response_context_items[1]
    assert isinstance(action, FinalAction)
    assert isinstance(stored_item["id"], str)
    assert isinstance(stored_item["status"], str)
    assert len(stored_item["id"]) < 700
    assert len(stored_item["status"]) < 700
    assert stored_item["id"].startswith("i" * 40)
    assert stored_item["status"].startswith("s" * 40)
    assert "[truncated" in stored_item["id"]
    assert "[truncated" in stored_item["status"]
    assert large_id not in str(state.response_context_items)
    assert large_status not in str(state.response_context_items)


def test_llm_client_bounds_large_nested_context_item_content(tmp_path) -> None:
    large_text = "x" * 10_000
    response = SimpleNamespace(
        id="resp-1",
        output_text="",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": large_text}],
            },
            {
                "type": "function_call",
                "name": "read_file",
                "arguments": '{"path": "README.md"}',
                "call_id": "call-1",
            },
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    content = state.response_context_items[1]["content"]
    assert isinstance(action, ToolCallAction)
    assert isinstance(content, list)
    stored_text = content[0]["text"]
    assert len(stored_text) < 4_200
    assert stored_text.startswith("x" * 40)
    assert "[truncated 5904 characters]" in stored_text
    assert large_text not in str(state.response_context_items)


def test_llm_client_bounds_wide_context_item_content(tmp_path) -> None:
    content = {
        "items": list(range(250)),
        **{f"key_{index}": index for index in range(250)},
    }
    response = SimpleNamespace(
        id="resp-1",
        output_text="done",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": content,
            }
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    stored_content = state.response_context_items[1]["content"]
    assert isinstance(action, FinalAction)
    assert isinstance(stored_content, dict)
    assert stored_content["key_0"] == 0
    assert stored_content["key_197"] == 197
    assert "key_198" not in stored_content
    assert "key_249" not in stored_content
    assert stored_content["__truncated_items__"] == {
        "truncated": True,
        "items": 251,
        "omitted": 52,
    }
    items = stored_content["items"]
    assert isinstance(items, list)
    assert items[198] == 198
    assert items[199] == {
        "truncated": True,
        "items": 250,
        "omitted": 51,
    }
    assert "key_249" not in str(state.response_context_items)


def test_llm_client_serializes_recursive_context_item_content(tmp_path) -> None:
    content = []
    content.append(content)
    response = SimpleNamespace(
        id="resp-1",
        output_text="done",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": content,
            }
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert state.response_context_items == [
        {"role": "user", "content": "read"},
        {
            "type": "message",
            "role": "assistant",
            "content": ["<recursive>"],
        },
    ]


def test_llm_client_serializes_uninspectable_context_item_content(tmp_path) -> None:
    class ExplodingItems(dict):
        def items(self):  # type: ignore[override]
            raise RuntimeError("items unavailable")

    response = SimpleNamespace(
        id="resp-1",
        output_text="done",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": ExplodingItems({"api_key": "sk-secret123"}),
            }
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert state.response_context_items == [
        {"role": "user", "content": "read"},
        {
            "type": "message",
            "role": "assistant",
            "content": "<uninspectable>",
        },
    ]
    assert "sk-secret123" not in str(state.response_context_items)


def test_llm_client_bounds_deep_context_item_content(tmp_path) -> None:
    content: dict[str, object] = {}
    current = content
    for _ in range(1_200):
        child: dict[str, object] = {}
        current["child"] = child
        current = child
    response = SimpleNamespace(
        id="resp-1",
        output_text="done",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": content,
            }
        ],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="read")

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
    ).next_action(state, ToolRegistry([DummyTool()]))

    assert isinstance(action, FinalAction)
    assert "<max-depth-exceeded>" in str(state.response_context_items)


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


def test_llm_client_serializes_unstringable_tool_observation_payloads(
    tmp_path,
) -> None:
    class UnstringableValue:
        def __str__(self) -> str:
            raise RuntimeError("value unavailable")

    class UnstringableKey:
        def __str__(self) -> str:
            raise RuntimeError("key unavailable")

    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeClient(response)
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="read",
        observations=[
            ToolObservation(
                call_id="call-1",
                tool_name="read_file",
                policy_decision="allow",
                result=ToolResult.model_construct(
                    success=True,
                    payload={UnstringableKey(): UnstringableValue()},
                    summary="ok",
                    error=None,
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
    assert observation_output["payload"] == {
        "<uninspectable>": "<uninspectable>"
    }


def test_llm_client_serializes_recursive_tool_observation_payloads(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeClient(response)
    payload = {}
    payload["self"] = payload
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="read",
        observations=[
            ToolObservation(
                call_id="call-1",
                tool_name="read_file",
                policy_decision="allow",
                result=ToolResult.model_construct(
                    success=True,
                    payload=payload,
                    summary="ok",
                    error=None,
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
    assert observation_output["payload"] == {"self": "<recursive>"}


def test_llm_client_bounds_deep_tool_observation_payloads(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeClient(response)
    payload: dict[str, object] = {}
    current = payload
    for _ in range(1_200):
        child: dict[str, object] = {}
        current["child"] = child
        current = child
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="read",
        observations=[
            ToolObservation(
                call_id="call-1",
                tool_name="read_file",
                policy_decision="allow",
                result=ToolResult.model_construct(
                    success=True,
                    payload=payload,
                    summary="ok",
                    error=None,
                ),
            )
        ],
    )

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=fake_client,
    ).next_action(state, ToolRegistry([DummyTool()]))

    output = fake_client.responses.calls[0]["input"][1]["output"]
    assert isinstance(action, FinalAction)
    assert "<max-depth-exceeded>" in output


def test_llm_client_bounds_wide_tool_observation_payloads(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeClient(response)
    payload = {
        "items": list(range(250)),
        **{f"key_{index}": index for index in range(250)},
    }
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="read",
        observations=[
            ToolObservation(
                call_id="call-1",
                tool_name="read_file",
                policy_decision="allow",
                result=ToolResult(success=True, payload=payload, summary="ok"),
            )
        ],
    )

    action = LLMClient(
        RunConfig(workspace=tmp_path),
        client=fake_client,
    ).next_action(state, ToolRegistry([DummyTool()]))

    output = fake_client.responses.calls[0]["input"][1]["output"]
    observation_output = json.loads(output)
    stored_payload = observation_output["payload"]
    assert isinstance(action, FinalAction)
    assert stored_payload["key_0"] == 0
    assert stored_payload["key_197"] == 197
    assert "key_198" not in stored_payload
    assert "key_249" not in stored_payload
    assert stored_payload["__truncated_items__"] == {
        "truncated": True,
        "items": 251,
        "omitted": 52,
    }
    assert stored_payload["items"][198] == 198
    assert stored_payload["items"][199] == {
        "truncated": True,
        "items": 250,
        "omitted": 51,
    }
    assert "key_249" not in output


def test_llm_client_bounds_large_tool_observation_outputs(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeClient(response)
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
                    payload={"content": "x" * 1_100_000},
                    summary="huge output",
                ),
            )
        ],
    )

    LLMClient(
        RunConfig(workspace=tmp_path),
        client=fake_client,
    ).next_action(state, ToolRegistry([DummyTool()]))

    output = fake_client.responses.calls[0]["input"][1]["output"]
    observation_output = json.loads(output)
    assert len(output.encode("utf-8")) < 2_000
    assert observation_output["payload"]["truncated"] is True
    assert observation_output["payload"]["bytes"] > 1_048_576
    assert observation_output["payload"]["preview"].startswith('{"success": true')
    assert "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" in observation_output["payload"][
        "preview"
    ]


def test_llm_client_bounds_large_tool_observation_summary(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeClient(response)
    summary = "x" * 1_100_000
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
                    payload={"content": "small"},
                    summary=summary,
                ),
            )
        ],
    )

    LLMClient(
        RunConfig(workspace=tmp_path),
        client=fake_client,
    ).next_action(state, ToolRegistry([DummyTool()]))

    output = fake_client.responses.calls[0]["input"][1]["output"]
    observation_output = json.loads(output)
    assert len(output.encode("utf-8")) < 2_000
    assert observation_output["summary"].startswith("x" * 40)
    assert observation_output["summary"].endswith("...")
    assert len(observation_output["summary"]) < 600
    assert observation_output["payload"]["truncated"] is True
    assert summary not in output


def test_llm_client_bounds_large_tool_observation_error(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="done", output=[])
    fake_client = FakeClient(response)
    error = "x" * 1_100_000
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
                    success=False,
                    payload={"content": "small"},
                    summary="failed",
                    error=error,
                ),
            )
        ],
    )

    LLMClient(
        RunConfig(workspace=tmp_path),
        client=fake_client,
    ).next_action(state, ToolRegistry([DummyTool()]))

    output = fake_client.responses.calls[0]["input"][1]["output"]
    observation_output = json.loads(output)
    assert len(output.encode("utf-8")) < 2_000
    assert observation_output["error"].startswith("x" * 40)
    assert observation_output["error"].endswith("...")
    assert len(observation_output["error"]) < 600
    assert observation_output["payload"]["truncated"] is True
    assert error not in output


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


def test_llm_client_bounds_oversized_transcript_context_to_recent_turns(
    tmp_path,
) -> None:
    response = SimpleNamespace(id="resp-2", output_text="continued", output=[])
    fake_client = FakeClient(response)
    state = RunState(
        run_id="run-2",
        workspace=tmp_path,
        goal="current request",
        conversation=[
            ConversationMessage(role="user", content=f"turn-{index}")
            for index in range(300)
        ],
    )

    LLMClient(
        RunConfig(workspace=tmp_path, session_context_turns=300),
        client=fake_client,
    ).next_action(
        state,
        ToolRegistry([DummyTool()]),
    )

    request_input = fake_client.responses.calls[0]["input"]
    serialized_input = json.dumps(request_input)
    assert len(request_input) == 200
    assert request_input[-1] == {"role": "user", "content": "current request"}
    assert "turn-0" not in serialized_input
    assert "turn-299" in serialized_input


def test_llm_client_bounds_large_transcript_turn_content(tmp_path) -> None:
    response = SimpleNamespace(id="resp-1", output_text="done", output=[])
    fake_client = FakeClient(response)
    large_content = "x" * 20_000
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="current",
        conversation=[
            ConversationMessage(role="user", content=large_content),
        ],
    )

    LLMClient(RunConfig(workspace=tmp_path), client=fake_client).next_action(
        state,
        ToolRegistry([DummyTool()]),
    )

    content = fake_client.responses.calls[0]["input"][0]["content"]
    assert len(content) < 5_000
    assert content.startswith("xxxxxxxxxxxxxxxx")
    assert "[truncated" in content
    assert large_content not in json.dumps(fake_client.responses.calls[0]["input"])


def test_llm_client_normalizes_oversized_transcript_roles(tmp_path) -> None:
    response = SimpleNamespace(id="resp-1", output_text="done", output=[])
    fake_client = FakeClient(response)
    oversized_role = "a" * 10_000
    state = RunState(
        run_id="run-1",
        workspace=tmp_path,
        goal="current",
        conversation=[
            ConversationMessage(role=oversized_role, content="remember alpha"),
        ],
    )

    LLMClient(RunConfig(workspace=tmp_path), client=fake_client).next_action(
        state,
        ToolRegistry([DummyTool()]),
    )

    request_input = fake_client.responses.calls[0]["input"]
    assert request_input[0]["role"] == "user"
    assert oversized_role not in json.dumps(request_input)


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


def test_llm_client_limits_sticky_context_turns(tmp_path) -> None:
    response = SimpleNamespace(id="resp-2", output_text="continued", output=[])
    fake_client = FakeClient(response)
    state = RunState(
        run_id="run-2",
        workspace=tmp_path,
        goal="current request",
        conversation=[
            ConversationMessage(
                role="user",
                content=f"sticky {index}",
                metadata={"sticky_context": True},
            )
            for index in range(4)
        ]
        + [
            ConversationMessage(role="user", content="recent user"),
            ConversationMessage(role="assistant", content="recent answer"),
        ],
    )

    LLMClient(
        RunConfig(workspace=tmp_path, session_context_turns=2),
        client=fake_client,
    ).next_action(state, ToolRegistry([DummyTool()]))

    call = fake_client.responses.calls[0]
    assert call["input"] == [
        {"role": "user", "content": "sticky 2"},
        {"role": "user", "content": "sticky 3"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "current request"},
    ]
