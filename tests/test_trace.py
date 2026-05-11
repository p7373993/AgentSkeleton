from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from agentskeleton.config import RunConfig
from agentskeleton.core.actions import FinalAction, ToolCallAction
from agentskeleton.core.llm import LLMClient
from agentskeleton.core.loop import AgentLoop
from agentskeleton.core.state import RunState
from agentskeleton.core.trace import ConsoleTraceSink, MemoryTraceSink
from agentskeleton.tools.base import Tool, ToolContext, ToolResult
from agentskeleton.tools.registry import ToolRegistry


class MemoryLogger:
    path = Path("memory.jsonl")

    def log(self, event_type: str, step: int, payload: dict[str, object]) -> None:
        pass


class ScriptedLLM:
    def __init__(self, actions) -> None:
        self.actions = list(actions)

    def next_action(self, state, registry):
        return self.actions.pop(0)


class TraceTool(Tool):
    name = "trace_tool"
    description = "Trace tool."
    risk = "read"
    args_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(success=True, payload={"ok": True}, summary="traced")


class FakeResponses:
    def __init__(self, response) -> None:
        self.response = response

    def create(self, **kwargs):
        return self.response


class FakeClient:
    def __init__(self, response) -> None:
        self.responses = FakeResponses(response)


def test_loop_emits_operational_trace(tmp_path: Path) -> None:
    trace = MemoryTraceSink()
    loop = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="trace_tool",
                    arguments={"value": "x"},
                    call_id="call-1",
                ),
                FinalAction(text="done"),
            ]
        ),
        registry=ToolRegistry([TraceTool()]),
        logger=MemoryLogger(),
        trace=trace,
    )

    loop.run("trace this")

    assert [event.name for event in trace.events] == [
        "run_started",
        "step_started",
        "model_action",
        "policy_decision",
        "tool_started",
        "tool_finished",
        "step_started",
        "run_finished",
    ]


def test_llm_client_emits_request_and_response_trace(tmp_path: Path) -> None:
    trace = MemoryTraceSink()
    response = SimpleNamespace(
        id="resp-1",
        output_text="done",
        output=[],
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="hello")

    LLMClient(
        RunConfig(workspace=tmp_path),
        client=FakeClient(response),
        trace=trace,
    ).next_action(state, ToolRegistry([TraceTool()]))

    assert [event.name for event in trace.events] == [
        "llm_request",
        "llm_response",
    ]
    assert trace.events[0].payload["input_preview"] == "hello"
    assert "registered function tools" in trace.events[0].payload[
        "instructions_preview"
    ]
    assert trace.events[0].payload["tool_names"] == ["trace_tool"]
    assert trace.events[1].payload["final_preview"] == "done"


def test_console_trace_sink_prints_readable_flow() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)

    trace.emit("tool_started", {"tool_name": "list_dir", "arguments": {"path": "."}})

    output = console.export_text()
    assert "[tool ->] list_dir" in output
    assert "path" in output


def test_console_trace_sink_prints_llm_instructions_preview() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)

    trace.emit(
        "llm_request",
        {
            "instructions_preview": "Use only registered tools.",
            "input_preview": "hello",
            "tool_names": ["read_file"],
        },
    )

    output = console.export_text()
    assert "[llm sys] Use only registered tools." in output
    assert "[llm ->]" in output


def test_console_trace_sink_prints_repeated_instructions_once() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    payload = {
        "instructions_preview": "Use only registered tools.",
        "input_preview": "hello",
        "tool_names": ["read_file"],
    }

    trace.emit("llm_request", payload)
    trace.emit("llm_request", payload)

    output = console.export_text()
    assert output.count("[llm sys]") == 1
    assert output.count("[llm ->]") == 2


def test_console_trace_sink_prints_repeated_large_instructions_once() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    payload = {
        "instructions_preview": "i" * 20_000,
        "input_preview": "hello",
        "tool_names": ["read_file"],
    }

    trace.emit("llm_request", payload)
    trace.emit("llm_request", payload)

    output = console.export_text()
    assert output.count("[llm sys]") == 1
    assert output.count("[llm ->]") == 2


def test_console_trace_sink_bounds_large_llm_tool_list() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    tool_names = [f"tool_{index}_{'t' * 100}" for index in range(128)]

    trace.emit(
        "llm_request",
        {
            "instructions_preview": "Use only registered tools.",
            "input_preview": "hello",
            "tool_names": tool_names,
        },
    )

    output = console.export_text()
    assert len(output) < 5_000
    assert "tool_0_" in output
    assert "..." in output


def test_console_trace_sink_bounds_large_llm_tool_call_list() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    calls = [
        {"name": f"tool_{index}_{'t' * 100}", "call_id": f"call_{index}"}
        for index in range(128)
    ]

    trace.emit("llm_response", {"function_calls": calls, "final_preview": None})

    output = console.export_text()
    assert len(output) < 5_000
    assert "tool_0_" in output
    assert "..." in output


def test_console_trace_sink_bounds_large_direct_llm_text_fields() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    large_text = "x" * 20_000

    trace.emit(
        "llm_request",
        {
            "instructions_preview": large_text,
            "input_preview": large_text,
            "tool_names": [],
        },
    )
    trace.emit(
        "llm_response",
        {
            "function_calls": [],
            "final_preview": large_text,
        },
    )

    output = console.export_text()
    assert len(output) < 5_000
    assert "xxxxxxxxxxxxxxxx" in output
    assert "..." in output
    assert large_text not in output


def test_console_trace_sink_bounds_large_policy_reason() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    reason = "r" * 20_000

    trace.emit(
        "policy_decision",
        {
            "tool_name": "trace_tool",
            "outcome": "confirm",
            "reason": reason,
        },
    )

    output = console.export_text()
    assert len(output) < 5_000
    assert "rrrrrrrrrrrrrrrr" in output
    assert "..." in output
    assert reason not in output


def test_console_trace_sink_redacts_secret_text_payloads() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)

    trace.emit(
        "tool_finished",
        {
            "tool_name": "trace_tool",
            "success": True,
            "summary": "OPENAI_API_KEY=sk-secret123",
        },
    )
    trace.emit(
        "llm_response",
        {
            "function_calls": [],
            "final_preview": "Authorization: Bearer token123",
        },
    )

    output = console.export_text()
    assert "sk-secret123" not in output
    assert "token123" not in output
    assert "[REDACTED]" in output


def test_console_trace_sink_serializes_recursive_payloads() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    arguments = {}
    arguments["self"] = arguments

    trace.emit("tool_started", {"tool_name": "trace_tool", "arguments": arguments})

    output = console.export_text()
    assert "<recursive>" in output


def test_console_trace_sink_serializes_uninspectable_payloads() -> None:
    class ExplodingItems(dict):
        def items(self):  # type: ignore[override]
            raise RuntimeError("items unavailable")

    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)

    trace.emit(
        "tool_started",
        {
            "tool_name": "trace_tool",
            "arguments": ExplodingItems({"api_key": "sk-secret123"}),
        },
    )

    output = console.export_text()
    assert "<uninspectable>" in output
    assert "sk-secret123" not in output


def test_console_trace_sink_redacts_secret_payloads() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)

    trace.emit(
        "tool_started",
        {
            "tool_name": "trace_tool",
            "arguments": {"api_key": "sk-secret123"},
        },
    )

    output = console.export_text()
    assert "sk-secret123" not in output
    assert "[REDACTED]" in output


def test_console_trace_sink_bounds_large_tool_arguments() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    arguments = {f"field_{index}": "a" * 100 for index in range(200)}

    trace.emit(
        "tool_started",
        {
            "tool_name": "trace_tool",
            "arguments": arguments,
        },
    )

    output = console.export_text()
    assert len(output) < 5_000
    assert "aaaaaaaaaaaaaaaa" in output
    assert "..." in output


def test_console_trace_sink_bounds_unknown_large_payload() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    payload = {f"field_{index}": "p" * 100 for index in range(200)}

    trace.emit("custom_event", {"payload": payload})

    output = console.export_text()
    assert len(output) < 5_000
    assert "pppppppppppppppp" in output
    assert "..." in output


def test_console_trace_sink_bounds_large_tool_summary() -> None:
    console = Console(record=True, width=120)
    trace = ConsoleTraceSink(console)
    large_summary = "s" * 20_000

    trace.emit(
        "tool_finished",
        {
            "tool_name": "trace_tool",
            "success": True,
            "summary": large_summary,
        },
    )

    output = console.export_text()
    assert len(output) < 5_000
    assert "ssssssssssssssss" in output
    assert "..." in output
    assert large_summary not in output
