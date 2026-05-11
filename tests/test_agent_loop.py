from pathlib import Path

import pytest

from agentskeleton.config import RunConfig
from agentskeleton.core.actions import FinalAction, ToolCallAction, ToolCallBatchAction
from agentskeleton.core.loop import AgentLoop
from agentskeleton.core.state import ConversationMessage
from agentskeleton.core.trace import MemoryTraceSink
from agentskeleton.tools.base import Tool, ToolContext, ToolResult
from agentskeleton.tools.registry import ToolRegistry
from agentskeleton.tools.user import AskUserTool


class MemoryLogger:
    def __init__(self) -> None:
        self.path = Path("memory.jsonl")
        self.events: list[tuple[str, int, dict[str, object]]] = []

    def log(self, event_type: str, step: int, payload: dict[str, object]) -> None:
        self.events.append((event_type, step, payload))


class FailingLogger:
    path = Path("failing.jsonl")

    def log(self, event_type: str, step: int, payload: dict[str, object]) -> None:
        raise OSError("log sink unavailable")


class FailingTrace:
    def emit(self, name: str, payload: dict[str, object]) -> None:
        raise OSError("trace sink unavailable")


class ScriptedLLM:
    def __init__(self, actions) -> None:
        self.actions = list(actions)

    def next_action(self, state, registry):
        return self.actions.pop(0)


class FailingLLM:
    def next_action(self, state, registry):
        raise RuntimeError("model unavailable")


class InvalidActionLLM:
    def next_action(self, state, registry):
        return {"type": "unexpected"}


class UnexpectedLLM:
    def next_action(self, state, registry):
        raise AssertionError("LLM should not be called")


class RecordTool(Tool):
    name = "record"
    description = "Record a value."
    risk = "read"
    args_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            payload={"value": args["value"], "workspace": str(context.workspace)},
            summary=f"recorded {args['value']}",
        )


class WriteRecordTool(RecordTool):
    name = "write_record"
    risk = "write"


class NullableRecordTool(RecordTool):
    name = "nullable_record"
    args_schema = {
        "type": "object",
        "properties": {"value": {"type": ["string", "null"]}},
        "required": ["value"],
        "additionalProperties": False,
    }


class ModeRecordTool(RecordTool):
    name = "mode_record"
    args_schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["fast", "safe"]}},
        "required": ["mode"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(success=True, payload={"mode": args["mode"]}, summary="ok")


class TagsRecordTool(RecordTool):
    name = "tags_record"
    args_schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["tags"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(success=True, payload={"tags": args["tags"]}, summary="ok")


class ConfigRecordTool(RecordTool):
    name = "config_record"
    args_schema = {
        "type": "object",
        "properties": {
            "settings": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
                "additionalProperties": False,
            }
        },
        "required": ["settings"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            payload={"settings": args["settings"]},
            summary="ok",
        )


class ObjectRecordTool(RecordTool):
    name = "object_record"
    args_schema = {
        "type": "object",
        "properties": {"payload": {"type": "object"}},
        "required": ["payload"],
        "additionalProperties": False,
    }

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        raise AssertionError("recursive arguments should not execute")


class ExplodingTool(RecordTool):
    name = "explode"

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        raise RuntimeError("boom")


class InvalidResultTool(RecordTool):
    name = "invalid_result"

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return {"success": True, "summary": "not a model"}  # type: ignore[return-value]


class MalformedResultTool(RecordTool):
    name = "malformed_result"

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult.model_construct(
            success="yes",
            payload=[],
            summary=123,
            error={},
        )


def make_loop(tmp_path: Path, actions, logger: MemoryLogger | None = None) -> AgentLoop:
    return AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(actions),
        registry=ToolRegistry([RecordTool()]),
        logger=logger or MemoryLogger(),
    )


def test_loop_stops_on_final_answer(tmp_path: Path) -> None:
    state = make_loop(tmp_path, [FinalAction(text="done")]).run("finish")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert state.step_count == 1


def test_loop_bounds_logged_final_answer(tmp_path: Path) -> None:
    logger = MemoryLogger()
    trace = MemoryTraceSink()
    answer = "x" * 5_000

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text=answer)]),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
        trace=trace,
    ).run("finish")

    logged_answer = next(
        event[2]["answer"] for event in logger.events if event[0] == "run_finished"
    )
    traced_answer = next(
        event.payload["answer"]
        for event in trace.events
        if event.name == "run_finished"
    )
    assert state.final_status == "completed"
    assert state.final_answer == answer
    assert isinstance(logged_answer, dict)
    assert logged_answer["truncated"] is True
    assert logged_answer["bytes"] == 5_000
    assert str(logged_answer["preview"]).startswith("xxxxxxxxxxxxxxxx")
    assert len(str(logged_answer["preview"])) < 300
    assert traced_answer == logged_answer
    assert answer not in str(logger.events)


def test_loop_rejects_blank_goal_before_model_call(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=UnexpectedLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("   ")

    assert state.final_status == "invalid_goal"
    assert state.final_reason == "Goal cannot be blank"
    assert state.final_answer is None
    assert state.step_count == 0
    assert not any(event[0] == "model_requested" for event in logger.events)
    assert logger.events[-1] == (
        "run_finished",
        0,
        {
            "status": "invalid_goal",
            "answer": None,
            "reason": "Goal cannot be blank",
        },
    )


def test_loop_logs_run_start_context(tmp_path: Path) -> None:
    logger = MemoryLogger()
    AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run(
        "continue",
        conversation=[{"role": "user", "content": "previous"}],
        trace_context={"session": "work"},
    )

    assert logger.events[0] == (
        "run_started",
        0,
        {
            "goal": "continue",
            "workspace": str(tmp_path),
            "resumed": True,
            "conversation_turns": 1,
            "session": "work",
        },
    )


def test_loop_trace_context_cannot_clobber_run_start_fields(tmp_path: Path) -> None:
    logger = MemoryLogger()
    trace = MemoryTraceSink()

    AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
        trace=trace,
    ).run(
        "continue",
        conversation=[{"role": "user", "content": "previous"}],
        trace_context={
            "goal": "spoofed",
            "workspace": "spoofed",
            "resumed": False,
            "conversation_turns": 999,
            "session": "work",
        },
    )

    assert logger.events[0] == (
        "run_started",
        0,
        {
            "goal": "continue",
            "workspace": str(tmp_path),
            "resumed": True,
            "conversation_turns": 1,
            "session": "work",
        },
    )
    assert trace.events[0].name == "run_started"
    assert trace.events[0].payload["goal"] == "continue"
    assert trace.events[0].payload["workspace"] == str(tmp_path)
    assert trace.events[0].payload["resumed"] is True
    assert trace.events[0].payload["conversation_turns"] == 1
    assert trace.events[0].payload["session"] == "work"


def test_loop_ignores_non_mapping_trace_context(tmp_path: Path) -> None:
    logger = MemoryLogger()

    try:
        state = AgentLoop(
            config=RunConfig(workspace=tmp_path),
            llm=ScriptedLLM([FinalAction(text="done")]),
            registry=ToolRegistry([RecordTool()]),
            logger=logger,
        ).run("finish", trace_context=["not", "a", "mapping"])  # type: ignore[arg-type]
    except Exception as exc:
        pytest.fail(f"loop raised instead of ignoring trace context: {exc!r}")

    assert state.final_status == "completed"
    assert logger.events[0] == (
        "run_started",
        0,
        {
            "goal": "finish",
            "workspace": str(tmp_path),
            "resumed": False,
            "conversation_turns": 0,
        },
    )


def test_loop_bounds_logged_trace_context_values(tmp_path: Path) -> None:
    logger = MemoryLogger()
    trace = MemoryTraceSink()
    session = "x" * 5_000

    AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
        trace=trace,
    ).run("finish", trace_context={"session": session, "attempt": 1})

    logged_payload = next(
        event[2] for event in logger.events if event[0] == "run_started"
    )
    traced_payload = next(
        event.payload for event in trace.events if event.name == "run_started"
    )
    assert isinstance(logged_payload["session"], dict)
    assert logged_payload["session"]["truncated"] is True
    assert logged_payload["session"]["bytes"] > 5_000
    assert logged_payload["attempt"] == 1
    assert traced_payload["session"] == logged_payload["session"]
    assert session not in str(logger.events)


def test_loop_normalizes_non_string_goal(tmp_path: Path) -> None:
    logger = MemoryLogger()
    seen_goals: list[str] = []

    class InspectingLLM:
        def next_action(self, state, registry):
            seen_goals.append(state.goal)
            return FinalAction(text="done")

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=InspectingLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run(123)

    assert state.final_status == "completed"
    assert state.goal == "123"
    assert seen_goals == ["123"]
    assert logger.events[0][2]["goal"] == "123"
    assert (
        "model_requested",
        1,
        {"goal": "123"},
    ) in logger.events


def test_loop_bounds_logged_goal_without_changing_model_input(tmp_path: Path) -> None:
    logger = MemoryLogger()
    trace = MemoryTraceSink()
    goal = "x" * 5_000
    seen_goals: list[str] = []

    class InspectingLLM:
        def next_action(self, state, registry):
            seen_goals.append(state.goal)
            return FinalAction(text="done")

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=InspectingLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
        trace=trace,
    ).run(goal)

    started_goal = next(
        event[2]["goal"] for event in logger.events if event[0] == "run_started"
    )
    requested_goal = next(
        event[2]["goal"] for event in logger.events if event[0] == "model_requested"
    )
    traced_goal = next(
        event.payload["goal"] for event in trace.events if event.name == "run_started"
    )
    assert state.goal == goal
    assert seen_goals == [goal]
    assert isinstance(started_goal, dict)
    assert started_goal["truncated"] is True
    assert started_goal["bytes"] == 5_000
    assert requested_goal == started_goal
    assert traced_goal == started_goal
    assert goal not in str(logger.events)


def test_loop_continues_when_logger_fails(tmp_path: Path) -> None:
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=FailingLogger(),
    ).run("finish")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert state.step_count == 1


def test_loop_continues_when_trace_sink_fails(tmp_path: Path) -> None:
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=MemoryLogger(),
        trace=FailingTrace(),
    ).run("finish")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert state.step_count == 1


def test_loop_logs_model_error_and_returns_state(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=FailingLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("finish")

    assert state.final_status == "model_error"
    assert state.final_reason == "Model call failed: RuntimeError"
    assert state.final_answer is None
    assert (
        "run_error",
        1,
        {
            "status": "model_error",
            "error_type": "RuntimeError",
            "error": "model unavailable",
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "model_error",
            "answer": None,
            "reason": "Model call failed: RuntimeError",
        },
    )


def test_loop_bounds_logged_model_error_message(tmp_path: Path) -> None:
    logger = MemoryLogger()
    trace = MemoryTraceSink()
    error_message = "x" * 5_000

    class LargeErrorLLM:
        def next_action(self, state, registry):
            raise RuntimeError(error_message)

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=LargeErrorLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
        trace=trace,
    ).run("finish")

    logged_error = next(
        event[2]["error"] for event in logger.events if event[0] == "run_error"
    )
    traced_error = next(
        event.payload["error"] for event in trace.events if event.name == "run_error"
    )
    assert state.final_status == "model_error"
    assert isinstance(logged_error, dict)
    assert logged_error["truncated"] is True
    assert logged_error["bytes"] == 5_000
    assert str(logged_error["preview"]).startswith("xxxxxxxxxxxxxxxx")
    assert len(str(logged_error["preview"])) < 300
    assert traced_error == logged_error
    assert error_message not in str(logger.events)


def test_loop_handles_unsupported_model_action_as_state(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=InvalidActionLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("finish")

    assert state.final_status == "invalid_action"
    assert state.final_reason == "Model returned unsupported action: dict"
    assert state.final_answer is None
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "dict",
            "error": "Model returned unsupported action: dict",
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "invalid_action",
            "answer": None,
            "reason": "Model returned unsupported action: dict",
        },
    )


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (
            FinalAction(text=None),  # type: ignore[arg-type]
            "Model returned invalid final action: text must be a string",
        ),
        (
            FinalAction(text="done", status=[]),  # type: ignore[arg-type]
            "Model returned invalid final action: status must be a non-empty string",
        ),
    ],
)
def test_loop_rejects_malformed_final_action_metadata(
    tmp_path: Path,
    action: FinalAction,
    reason: str,
) -> None:
    logger = MemoryLogger()
    state = make_loop(tmp_path, [action], logger).run("finish")

    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
    assert state.final_answer is None
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "invalid_final_action",
            "error": reason,
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "invalid_action",
            "answer": None,
            "reason": reason,
        },
    )


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (
            ToolCallAction(
                tool_name=[],
                arguments={"value": "x"},
                call_id="call-1",
            ),
            "Model returned invalid tool call: tool_name must be a non-empty string",
        ),
        (
            ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id=[],
            ),
            "Model returned invalid tool call: call_id must be a non-empty string",
        ),
        (
            ToolCallAction(
                tool_name="a" * 513,
                arguments={"value": "x"},
                call_id="call-1",
            ),
            "Model returned invalid tool call: tool_name exceeds 512 bytes",
        ),
        (
            ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id="a" * 513,
            ),
            "Model returned invalid tool call: call_id exceeds 512 bytes",
        ),
    ],
)
def test_loop_rejects_malformed_tool_call_metadata(
    tmp_path: Path,
    action: ToolCallAction,
    reason: str,
) -> None:
    logger = MemoryLogger()
    try:
        state = make_loop(tmp_path, [action], logger).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid action: {exc!r}")

    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
    assert state.final_answer is None
    assert state.observations == []
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "invalid_tool_call",
            "error": reason,
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "invalid_action",
            "answer": None,
            "reason": reason,
        },
    )
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_executes_tool_and_feeds_observation(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id="call-1",
            ),
            FinalAction(text="done"),
        ],
        logger,
    ).run("record")

    assert state.final_status == "completed"
    assert len(state.observations) == 1
    assert state.observations[0].call_id == "call-1"
    assert state.observations[0].result.success is True
    assert any(event[0] == "tool_finished" for event in logger.events)


def test_loop_bounds_logged_tool_arguments_without_changing_execution(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    trace = MemoryTraceSink()
    large_value = "x" * 5_000
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="record",
                    arguments={"value": large_value},
                    call_id="call-1",
                ),
                FinalAction(text="done"),
            ]
        ),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
        trace=trace,
    ).run("record")

    model_action = next(event for event in logger.events if event[0] == "model_action")
    trace_arguments = next(
        event.payload["arguments"]
        for event in trace.events
        if event.name == "tool_started"
    )
    assert state.final_status == "completed"
    assert state.observations[0].result.payload["value"] == large_value
    assert model_action[2]["arguments"] == {
        "truncated": True,
        "bytes": 5012,
        "preview": '{"value":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...',
    }
    assert trace_arguments == model_action[2]["arguments"]


def test_loop_executes_all_tool_calls_in_batch(tmp_path: Path) -> None:
    state = make_loop(
        tmp_path,
        [
            ToolCallBatchAction(
                tool_calls=[
                    ToolCallAction(
                        tool_name="record",
                        arguments={"value": "x"},
                        call_id="call-1",
                    ),
                    ToolCallAction(
                        tool_name="record",
                        arguments={"value": "y"},
                        call_id="call-2",
                    ),
                ]
            ),
            FinalAction(text="done"),
        ],
    ).run("record")

    assert state.final_status == "completed"
    assert [observation.call_id for observation in state.observations] == [
        "call-1",
        "call-2",
    ]
    values = [
        observation.result.payload["value"] for observation in state.observations
    ]
    assert values == [
        "x",
        "y",
    ]


def test_loop_handles_empty_tool_call_batch_as_invalid_action(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [ToolCallBatchAction(tool_calls=[])],
        logger,
    ).run("record")

    assert state.final_status == "invalid_action"
    assert state.final_reason == "Model returned empty tool call batch"
    assert state.final_answer is None
    assert state.observations == []
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "empty_batch",
            "error": "Model returned empty tool call batch",
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "invalid_action",
            "answer": None,
            "reason": "Model returned empty tool call batch",
        },
    )


def test_loop_rejects_duplicate_batch_call_ids_before_execution(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallBatchAction(
                tool_calls=[
                    ToolCallAction(
                        tool_name="record",
                        arguments={"value": "x"},
                        call_id="call-1",
                    ),
                    ToolCallAction(
                        tool_name="record",
                        arguments={"value": "y"},
                        call_id="call-1",
                    ),
                ]
            )
        ],
        logger,
    ).run("record")

    reason = "Model returned invalid tool call batch: duplicate call_id: call-1"
    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
    assert state.final_answer is None
    assert state.observations == []
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "invalid_tool_batch",
            "error": reason,
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "invalid_action",
            "answer": None,
            "reason": reason,
        },
    )
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_oversized_tool_call_batch_before_execution(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    tool_calls = [
        ToolCallAction(
            tool_name="record",
            arguments={"value": str(index)},
            call_id=f"call-{index}",
        )
        for index in range(21)
    ]

    state = make_loop(
        tmp_path,
        [ToolCallBatchAction(tool_calls=tool_calls)],
        logger,
    ).run("record")

    reason = "Model returned invalid tool call batch: too many tool calls (max 20)"
    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
    assert state.final_answer is None
    assert state.observations == []
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "invalid_tool_batch",
            "error": reason,
        },
    ) in logger.events
    assert not any(event[0] == "tool_started" for event in logger.events)


@pytest.mark.parametrize("tool_calls", [None, 123, "record"])
def test_loop_rejects_malformed_tool_call_batch_shape(
    tmp_path: Path,
    tool_calls: object,
) -> None:
    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [ToolCallBatchAction(tool_calls=tool_calls)],  # type: ignore[arg-type]
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid batch: {exc!r}")

    reason = "Model returned invalid tool call batch: tool_calls must be a list"
    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
    assert state.final_answer is None
    assert state.observations == []
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "invalid_tool_batch",
            "error": reason,
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "invalid_action",
            "answer": None,
            "reason": reason,
        },
    )
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_invalid_batch_member_before_executing_earlier_calls(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallBatchAction(
                tool_calls=[
                    ToolCallAction(
                        tool_name="record",
                        arguments={"value": "x"},
                        call_id="call-1",
                    ),
                    {"type": "unexpected"},
                ]
            )
        ],
        logger,
    ).run("record")

    reason = "Model returned unsupported action: dict"
    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
    assert state.final_answer is None
    assert state.observations == []
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "dict",
            "error": reason,
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "invalid_action",
            "answer": None,
            "reason": reason,
        },
    )
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_stops_at_max_steps(tmp_path: Path) -> None:
    logger = MemoryLogger()
    loop = AgentLoop(
        config=RunConfig(workspace=tmp_path, max_steps=1),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="record",
                    arguments={"value": "x"},
                    call_id="call-1",
                ),
                FinalAction(text="unreachable"),
            ]
        ),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    )

    state = loop.run("record")

    assert state.final_status == "max_steps"
    assert state.final_answer is None
    assert state.final_reason == "Reached max_steps limit: 1"
    assert state.step_count == 1
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "max_steps",
            "answer": None,
            "reason": "Reached max_steps limit: 1",
        },
    )


def test_loop_stops_when_policy_blocks_tool(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="write_record",
                    arguments={"value": "x"},
                    call_id="call-1",
                )
            ]
        ),
        registry=ToolRegistry([WriteRecordTool()]),
        logger=logger,
        confirmer=lambda decision, action: False,
    ).run("record")

    assert state.final_status == "denied"
    assert state.final_reason == "Tool writes files in the workspace"
    assert state.observations[0].policy_decision == "confirm"
    assert (
        "tool_finished",
        1,
        {
            "tool_name": "write_record",
            "success": False,
            "summary": "Tool writes files in the workspace",
            "error": "Permission denied",
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "denied",
            "answer": None,
            "reason": "Tool writes files in the workspace",
        },
    )


def test_loop_denies_tool_when_confirmer_raises(tmp_path: Path) -> None:
    logger = MemoryLogger()

    def failing_confirmer(decision, action) -> bool:
        raise RuntimeError("input stream closed")

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="write_record",
                    arguments={"value": "x"},
                    call_id="call-1",
                )
            ]
        ),
        registry=ToolRegistry([WriteRecordTool()]),
        logger=logger,
        confirmer=failing_confirmer,
    ).run("record")

    assert state.final_status == "denied"
    assert state.final_reason == "Permission confirmation failed: RuntimeError"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "confirm"
    assert observation.result.success is False
    assert observation.result.summary == "Permission confirmation failed: RuntimeError"
    assert observation.result.error == "input stream closed"
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "denied",
            "answer": None,
            "reason": "Permission confirmation failed: RuntimeError",
        },
    )


def test_loop_denies_tool_when_confirmer_returns_non_boolean(tmp_path: Path) -> None:
    logger = MemoryLogger()

    def invalid_confirmer(decision, action):
        return "yes"

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="write_record",
                    arguments={"value": "x"},
                    call_id="call-1",
                ),
                FinalAction(text="unreachable"),
            ]
        ),
        registry=ToolRegistry([WriteRecordTool()]),
        logger=logger,
        confirmer=invalid_confirmer,
    ).run("record")

    expected_reason = "Permission confirmation returned invalid result: str"
    assert state.final_status == "denied"
    assert state.final_reason == expected_reason
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "confirm"
    assert observation.result.success is False
    assert observation.result.summary == expected_reason
    assert observation.result.error == "Permission confirmation invalid"
    assert not any(event[0] == "tool_started" for event in logger.events)
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "denied",
            "answer": None,
            "reason": expected_reason,
        },
    )


def test_loop_uses_permission_profile_from_config(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path, permission_profile="read_only"),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="write_record",
                    arguments={"value": "x"},
                    call_id="call-1",
                )
            ]
        ),
        registry=ToolRegistry([WriteRecordTool()]),
        logger=logger,
        confirmer=lambda decision, action: True,
    ).run("record")

    assert state.final_status == "blocked"
    assert state.final_reason == "Tool is blocked by read-only profile"
    assert state.observations[0].policy_decision == "block"
    assert state.observations[0].result.error == "Permission blocked"
    assert (
        "tool_finished",
        1,
        {
            "tool_name": "write_record",
            "success": False,
            "summary": "Tool is blocked by read-only profile",
            "error": "Permission blocked",
        },
    ) in logger.events
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "blocked",
            "answer": None,
            "reason": "Tool is blocked by read-only profile",
        },
    )


def test_loop_stops_with_unknown_tool_status(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name="missing",
                arguments={"value": "x"},
                call_id="call-1",
            )
        ],
        logger,
    ).run("record")

    assert state.final_status == "unknown_tool"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.tool_name == "missing"
    assert observation.policy_decision == "block"
    assert observation.result.success is False
    assert observation.result.summary == "Unknown tool: missing"
    assert observation.result.error == "Unknown tool"
    assert observation.result.payload == {
        "tool_name": "missing",
        "arguments": {"value": "x"},
    }
    assert state.final_reason == "Unknown tool: missing"
    assert (
        "tool_finished",
        1,
        {
            "tool_name": "missing",
            "success": False,
            "summary": "Unknown tool: missing",
            "error": "Unknown tool",
        },
    ) in logger.events


@pytest.mark.parametrize(
    ("arguments", "validation_errors"),
    [
        ({}, ["Missing required argument: value"]),
        ({"value": 123}, ["Argument value must be string"]),
        (
            {"value": "x", "extra": "y"},
            ["Unexpected argument: extra"],
        ),
    ],
)
def test_loop_rejects_invalid_tool_arguments_before_execution(
    tmp_path: Path,
    arguments: dict[str, object],
    validation_errors: list[str],
) -> None:
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name="record",
                arguments=arguments,
                call_id="call-1",
            ),
            FinalAction(text="recovered"),
        ],
        logger,
    ).run("record")

    assert state.final_status == "completed"
    assert state.final_answer == "recovered"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "block"
    assert observation.result.success is False
    assert observation.result.summary == "Invalid tool arguments"
    assert observation.result.error == "Invalid arguments"
    assert observation.result.payload == {
        "tool_name": "record",
        "arguments": arguments,
        "validation_errors": validation_errors,
    }
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_bounds_deep_tool_arguments_before_validation(tmp_path: Path) -> None:
    logger = MemoryLogger()
    value: dict[str, object] = {}
    current = value
    for _ in range(20_000):
        child: dict[str, object] = {}
        current["child"] = child
        current = child

    state = make_loop(
        tmp_path,
        [
            ToolCallAction("record", {"value": value}, "call-1"),
            FinalAction(text="recovered"),
        ],
        logger,
    ).run("record")

    assert state.final_status == "completed"
    assert state.final_answer == "recovered"
    assert state.observations[0].result.error == "Invalid arguments"
    model_action = next(event for event in logger.events if event[0] == "model_action")
    assert "<max-depth-exceeded>" in str(model_action[2]["arguments"])


def test_loop_rejects_recursive_tool_arguments_before_execution(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    payload: dict[str, object] = {}
    payload["self"] = payload
    loop = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction("object_record", {"payload": payload}, "call-1"),
                FinalAction(text="recovered"),
            ]
        ),
        registry=ToolRegistry([ObjectRecordTool()]),
        logger=logger,
    )

    state = loop.run("record object")

    assert state.final_status == "completed"
    assert state.final_answer == "recovered"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.result.error == "Invalid arguments"
    assert observation.result.payload["validation_errors"] == [
        "Tool arguments cannot contain recursive values"
    ]
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_non_object_tool_arguments_before_execution(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name="record",
                arguments=[],
                call_id="call-1",
            ),
            FinalAction(text="recovered"),
        ],
        logger,
    ).run("record")

    assert state.final_status == "completed"
    assert state.final_answer == "recovered"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "block"
    assert observation.result.success is False
    assert observation.result.payload == {
        "tool_name": "record",
        "arguments": [],
        "validation_errors": ["Tool arguments must be an object"],
    }
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_non_string_tool_argument_names_before_execution(
    tmp_path: Path,
) -> None:
    bad_key = object()
    arguments = {bad_key: "x"}
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name="record",
                arguments=arguments,
                call_id="call-1",
            ),
            FinalAction(text="recovered"),
        ],
        logger,
    ).run("record")

    assert state.final_status == "completed"
    assert state.final_answer == "recovered"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "block"
    assert observation.result.success is False
    assert observation.result.payload == {
        "tool_name": "record",
        "arguments": arguments,
        "validation_errors": ["Tool argument names must be strings"],
    }
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_invalid_union_type_tool_argument(tmp_path: Path) -> None:
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="nullable_record",
                    arguments={"value": 123},
                    call_id="call-1",
                ),
                FinalAction(text="recovered"),
            ]
        ),
        registry=ToolRegistry([NullableRecordTool()]),
        logger=MemoryLogger(),
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].result.success is False
    assert state.observations[0].result.payload["validation_errors"] == [
        "Argument value must be string or null"
    ]


def test_loop_rejects_tool_arguments_outside_enum(tmp_path: Path) -> None:
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="mode_record",
                    arguments={"mode": "turbo"},
                    call_id="call-1",
                ),
                FinalAction(text="recovered"),
            ]
        ),
        registry=ToolRegistry([ModeRecordTool()]),
        logger=MemoryLogger(),
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].result.success is False
    assert state.observations[0].result.payload["validation_errors"] == [
        "Argument mode must be one of: fast, safe"
    ]


def test_loop_rejects_invalid_array_item_tool_argument(tmp_path: Path) -> None:
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="tags_record",
                    arguments={"tags": ["ok", 123]},
                    call_id="call-1",
                ),
                FinalAction(text="recovered"),
            ]
        ),
        registry=ToolRegistry([TagsRecordTool()]),
        logger=MemoryLogger(),
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].result.success is False
    assert state.observations[0].result.payload["validation_errors"] == [
        "Argument tags[1] must be string"
    ]


def test_loop_rejects_invalid_nested_object_tool_argument(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="config_record",
                    arguments={"settings": {"enabled": "yes"}},
                    call_id="call-1",
                ),
                FinalAction(text="recovered"),
            ]
        ),
        registry=ToolRegistry([ConfigRecordTool()]),
        logger=logger,
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].result.success is False
    assert state.observations[0].result.payload["validation_errors"] == [
        "Argument settings.enabled must be boolean"
    ]
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_invalid_nested_object_shape_tool_argument(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="config_record",
                    arguments={"settings": {"extra": True}},
                    call_id="call-1",
                ),
                FinalAction(text="recovered"),
            ]
        ),
        registry=ToolRegistry([ConfigRecordTool()]),
        logger=logger,
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].result.success is False
    assert state.observations[0].result.payload["validation_errors"] == [
        "Missing required argument: settings.enabled",
        "Unexpected argument: settings.extra",
    ]
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_converts_tool_exception_to_observation(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="explode",
                    arguments={"value": "x"},
                    call_id="call-1",
                )
            ]
        ),
        registry=ToolRegistry([ExplodingTool()]),
        logger=logger,
    ).run("record")

    assert state.final_status == "tool_error"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "allow"
    assert observation.result.success is False
    assert observation.result.summary == "Tool raised an exception: RuntimeError"
    assert observation.result.error == "boom"
    assert observation.result.payload == {
        "tool_name": "explode",
        "arguments": {"value": "x"},
    }
    assert (
        "tool_finished",
        1,
        {
            "tool_name": "explode",
            "success": False,
            "summary": "Tool raised an exception: RuntimeError",
            "error": "boom",
        },
    ) in logger.events


def test_loop_bounds_logged_tool_error_message(tmp_path: Path) -> None:
    logger = MemoryLogger()
    trace = MemoryTraceSink()
    error_message = "x" * 5_000

    class LargeErrorTool(RecordTool):
        name = "large_error"

        def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
            raise RuntimeError(error_message)

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="large_error",
                    arguments={"value": "x"},
                    call_id="call-1",
                )
            ]
        ),
        registry=ToolRegistry([LargeErrorTool()]),
        logger=logger,
        trace=trace,
    ).run("record")

    logged_error = next(
        event[2]["error"] for event in logger.events if event[0] == "tool_finished"
    )
    traced_error = next(
        event.payload["error"]
        for event in trace.events
        if event.name == "tool_finished"
    )
    assert state.final_status == "tool_error"
    assert state.observations[0].result.error == error_message
    assert isinstance(logged_error, dict)
    assert logged_error["truncated"] is True
    assert logged_error["bytes"] == 5_000
    assert str(logged_error["preview"]).startswith("xxxxxxxxxxxxxxxx")
    assert len(str(logged_error["preview"])) < 300
    assert traced_error == logged_error
    assert error_message not in str(logger.events)


def test_loop_converts_invalid_tool_result_to_observation(tmp_path: Path) -> None:
    logger = MemoryLogger()
    try:
        state = AgentLoop(
            config=RunConfig(workspace=tmp_path),
            llm=ScriptedLLM(
                [
                    ToolCallAction(
                        tool_name="invalid_result",
                        arguments={"value": "x"},
                        call_id="call-1",
                    )
                ]
            ),
            registry=ToolRegistry([InvalidResultTool()]),
            logger=logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording a tool error: {exc!r}")

    assert state.final_status == "tool_error"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "allow"
    assert observation.result.success is False
    assert observation.result.summary == "Tool returned invalid result: dict"
    assert observation.result.error == "Invalid tool result"
    assert observation.result.payload == {
        "tool_name": "invalid_result",
        "arguments": {"value": "x"},
        "result_type": "dict",
    }
    assert (
        "tool_finished",
        1,
        {
            "tool_name": "invalid_result",
            "success": False,
            "summary": "Tool returned invalid result: dict",
            "error": "Invalid tool result",
        },
    ) in logger.events


def test_loop_converts_malformed_tool_result_to_observation(tmp_path: Path) -> None:
    logger = MemoryLogger()
    try:
        state = AgentLoop(
            config=RunConfig(workspace=tmp_path),
            llm=ScriptedLLM(
                [
                    ToolCallAction(
                        tool_name="malformed_result",
                        arguments={"value": "x"},
                        call_id="call-1",
                    )
                ]
            ),
            registry=ToolRegistry([MalformedResultTool()]),
            logger=logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording a tool error: {exc!r}")

    assert state.final_status == "tool_error"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "allow"
    assert observation.result.success is False
    assert observation.result.summary == (
        "Tool returned malformed result: success must be a boolean"
    )
    assert observation.result.error == "Malformed tool result"
    assert observation.result.payload == {
        "tool_name": "malformed_result",
        "arguments": {"value": "x"},
        "validation_errors": [
            "success must be a boolean",
            "payload must be a mapping",
            "summary must be a string",
            "error must be a string or null",
        ],
    }
    assert (
        "tool_finished",
        1,
        {
            "tool_name": "malformed_result",
            "success": False,
            "summary": "Tool returned malformed result: success must be a boolean",
            "error": "Malformed tool result",
        },
    ) in logger.events


def test_loop_stops_after_repeating_same_action_three_times(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id="call-1",
            ),
            ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id="call-2",
            ),
            ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id="call-3",
            ),
            FinalAction(text="unreachable"),
        ],
        logger,
    ).run("record")

    assert state.final_status == "repeated_action"
    assert state.step_count == 3
    assert state.repeated_action_count == 3
    assert [observation.result.success for observation in state.observations] == [
        True,
        True,
        False,
    ]
    assert state.observations[2].result.error == "Repeated action"
    assert (
        "tool_finished",
        3,
        {
            "tool_name": "record",
            "success": False,
            "summary": "Repeated tool action 3 times: record",
            "error": "Repeated action",
        },
    ) in logger.events


def test_loop_resets_repeat_counter_when_arguments_change(tmp_path: Path) -> None:
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id="call-1",
            ),
            ToolCallAction(
                tool_name="record",
                arguments={"value": "y"},
                call_id="call-2",
            ),
            ToolCallAction(
                tool_name="record",
                arguments={"value": "y"},
                call_id="call-3",
            ),
            FinalAction(text="done"),
        ],
    ).run("record")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert state.repeated_action_count == 2
    values = [
        observation.result.payload["value"] for observation in state.observations
    ]
    assert values == [
        "x",
        "y",
        "y",
    ]


def test_loop_uses_bounded_repeated_action_fingerprints(tmp_path: Path) -> None:
    large_value = "x" * 5_000
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name="record",
                arguments={"value": large_value},
                call_id="call-1",
            ),
            ToolCallAction(
                tool_name="record",
                arguments={"value": large_value},
                call_id="call-2",
            ),
            FinalAction(text="done"),
        ],
    ).run("record")

    assert state.final_status == "completed"
    assert state.repeated_action_count == 2
    assert state.last_action_fingerprint is not None
    assert len(state.last_action_fingerprint) <= 96
    assert large_value not in state.last_action_fingerprint


def test_loop_passes_user_input_callback_to_tools(tmp_path: Path) -> None:
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="ask_user",
                    arguments={"question": "Continue?"},
                    call_id="call-1",
                ),
                FinalAction(text="done"),
            ]
        ),
        registry=ToolRegistry([AskUserTool()]),
        logger=MemoryLogger(),
        ask_user=lambda question: f"yes: {question}",
    ).run("ask")

    assert state.final_status == "completed"
    assert state.observations[0].result.payload["answer"] == "yes: Continue?"


def test_loop_starts_with_conversation_history(tmp_path: Path) -> None:
    seen_conversation: list[list[str]] = []

    class InspectingLLM:
        def next_action(self, state, registry):
            seen_conversation.append([turn.content for turn in state.conversation])
            return FinalAction(text="done")

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=InspectingLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=MemoryLogger(),
    ).run(
        "continue",
        conversation=[
            {"role": "user", "content": "remember alpha"},
            {"role": "assistant", "content": "alpha stored"},
        ],
    )

    assert state.final_status == "completed"
    assert seen_conversation == [["remember alpha", "alpha stored"]]


def test_loop_treats_scalar_conversation_as_single_user_turn(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    seen_conversation: list[list[tuple[str, str, dict[str, object]]]] = []

    class InspectingLLM:
        def next_action(self, state, registry):
            seen_conversation.append(
                [
                    (turn.role, turn.content, turn.metadata)
                    for turn in state.conversation
                ]
            )
            return FinalAction(text="done")

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=InspectingLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run(
        "continue",
        conversation="loose history entry",  # type: ignore[arg-type]
    )

    assert state.final_status == "completed"
    assert seen_conversation == [[("user", "loose history entry", {})]]
    assert logger.events[0][2]["resumed"] is True
    assert logger.events[0][2]["conversation_turns"] == 1


def test_loop_normalizes_non_mapping_conversation_entries(tmp_path: Path) -> None:
    logger = MemoryLogger()
    seen_conversation: list[list[tuple[str, str, dict[str, object]]]] = []

    class InspectingLLM:
        def next_action(self, state, registry):
            seen_conversation.append(
                [
                    (turn.role, turn.content, turn.metadata)
                    for turn in state.conversation
                ]
            )
            return FinalAction(text="done")

    try:
        state = AgentLoop(
            config=RunConfig(workspace=tmp_path),
            llm=InspectingLLM(),
            registry=ToolRegistry([RecordTool()]),
            logger=logger,
        ).run(
            "continue",
            conversation=[
                "loose history entry",
                {"role": "assistant", "content": "structured entry"},
            ],
        )
    except Exception as exc:
        pytest.fail(f"loop raised instead of normalizing conversation: {exc!r}")

    assert state.final_status == "completed"
    assert seen_conversation == [
        [
            ("user", "loose history entry", {}),
            ("assistant", "structured entry", {}),
        ]
    ]
    assert logger.events[0][2]["resumed"] is True
    assert logger.events[0][2]["conversation_turns"] == 2


def test_loop_sanitizes_conversation_message_entries(tmp_path: Path) -> None:
    seen_conversation: list[list[tuple[str, str, dict[str, object]]]] = []

    class InspectingLLM:
        def next_action(self, state, registry):
            seen_conversation.append(
                [
                    (turn.role, turn.content, turn.metadata)
                    for turn in state.conversation
                ]
            )
            return FinalAction(text="done")

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=InspectingLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=MemoryLogger(),
    ).run(
        "continue",
        conversation=[
            ConversationMessage(
                role=123,  # type: ignore[arg-type]
                content=None,  # type: ignore[arg-type]
                metadata=["bad"],  # type: ignore[arg-type]
            ),
        ],
    )

    assert state.final_status == "completed"
    assert seen_conversation == [[("123", "None", {})]]
