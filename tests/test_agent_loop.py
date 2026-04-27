from pathlib import Path

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


def test_loop_stops_at_max_steps(tmp_path: Path) -> None:
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
        logger=MemoryLogger(),
    )

    state = loop.run("record")

    assert state.final_status == "max_steps"
    assert state.final_answer is None
    assert state.step_count == 1


def test_loop_stops_when_policy_blocks_tool(tmp_path: Path) -> None:
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
        logger=MemoryLogger(),
        confirmer=lambda decision, action: False,
    ).run("record")

    assert state.final_status == "denied"
    assert state.observations[0].policy_decision == "confirm"


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
