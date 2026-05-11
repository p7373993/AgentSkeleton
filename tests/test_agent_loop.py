from pathlib import Path

import pytest

from agentskeleton.config import RunConfig
from agentskeleton.core.actions import FinalAction, ToolCallAction, ToolCallBatchAction
from agentskeleton.core.loop import AgentLoop
from agentskeleton.tools.base import Tool, ToolContext, ToolResult
from agentskeleton.tools.registry import ToolRegistry
from agentskeleton.tools.user import AskUserTool


class MemoryLogger:
    def __init__(self) -> None:
        self.path = Path("memory.jsonl")
        self.events: list[tuple[str, int, dict[str, object]]] = []

    def log(self, event_type: str, step: int, payload: dict[str, object]) -> None:
        self.events.append((event_type, step, payload))


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


class ExplodingTool(RecordTool):
    name = "explode"

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        raise RuntimeError("boom")


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
