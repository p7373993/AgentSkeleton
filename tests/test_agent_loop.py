import hashlib
import json
from pathlib import Path

import pytest

from agentskeleton.config import RunConfig
from agentskeleton.core.actions import FinalAction, ToolCallAction, ToolCallBatchAction
from agentskeleton.core.loop import AgentLoop
from agentskeleton.core.state import ConversationMessage, RunState
from agentskeleton.core.trace import MemoryTraceSink
from agentskeleton.policy.permissions import PermissionDecision, PermissionPolicy
from agentskeleton.tools.base import Tool, ToolContext, ToolResult
from agentskeleton.tools.registry import ToolRegistry
from agentskeleton.tools.user import AskUserTool


def schema_hash(schema: dict[str, object]) -> str:
    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tool_implementation(tool_type: type[Tool]) -> str:
    return f"{tool_type.__module__}.{tool_type.__qualname__}"


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


class UninspectableResultTool(RecordTool):
    name = "uninspectable_result"

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        class UninspectableResult(ToolResult):
            def __getattribute__(self, name: str):
                if name == "success":
                    raise RuntimeError("success unavailable")
                return super().__getattribute__(name)

        return UninspectableResult(
            success=True,
            payload={"value": args["value"]},
            summary="ok",
        )


class FlakyResultTool(RecordTool):
    name = "flaky_result"

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        reads = {"success": 0}

        class FlakyResult(ToolResult):
            def __getattribute__(self, name: str):
                if name == "success":
                    reads["success"] += 1
                    if reads["success"] > 2:
                        raise RuntimeError("success changed after validation")
                return super().__getattribute__(name)

        return FlakyResult(
            success=True,
            payload={"value": args["value"]},
            summary="flaky ok",
        )


class LargePayloadTool(RecordTool):
    name = "large_payload"

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            payload={"content": "x" * 1_100_000},
            summary="large payload ok",
        )


class LargeSummaryTool(RecordTool):
    name = "large_summary"

    def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            payload={"ok": True},
            summary="s" * 1_100_000,
        )


def make_loop(tmp_path: Path, actions, logger: MemoryLogger | None = None) -> AgentLoop:
    return AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(actions),
        registry=ToolRegistry([RecordTool()]),
        logger=logger or MemoryLogger(),
    )


def default_run_start_runtime() -> dict[str, object]:
    return {
        "model": "gpt-5.5",
        "reasoning_effort": "low",
        "text_verbosity": "low",
        "max_steps": 20,
        "model_retry_attempts": 2,
        "permission_profile": "standard",
        "confirm_risky_actions": True,
        "enabled_tools": None,
        "tool_modules": [],
        "registered_tools": [
            {
                "name": "record",
                "description": "Record a value.",
                "risk": "read",
                "args_schema_hash": schema_hash(RecordTool.args_schema),
                "implementation": tool_implementation(RecordTool),
            }
        ],
    }


def test_loop_stops_on_final_answer(tmp_path: Path) -> None:
    state = make_loop(tmp_path, [FinalAction(text="done")]).run("finish")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert state.step_count == 1


def test_loop_preserves_truthless_trace_sink(tmp_path: Path) -> None:
    class TruthlessTrace(MemoryTraceSink):
        def __bool__(self) -> bool:
            raise RuntimeError("trace truthiness unavailable")

    trace = TruthlessTrace()

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=MemoryLogger(),
        trace=trace,
    ).run("finish")

    assert state.final_status == "completed"
    assert [event.name for event in trace.events][:2] == [
        "run_started",
        "step_started",
    ]


def test_loop_preserves_truthless_policy(tmp_path: Path) -> None:
    class TruthlessPolicy(PermissionPolicy):
        def __bool__(self) -> bool:
            raise RuntimeError("policy truthiness unavailable")

        def decide(
            self,
            tool_name: str,
            args: dict[str, object],
            risk: str,
        ) -> PermissionDecision:
            return PermissionDecision("block", "truthless policy used")

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="record",
                    call_id="record-1",
                    arguments={"value": "hello"},
                )
            ]
        ),
        registry=ToolRegistry([RecordTool()]),
        logger=MemoryLogger(),
        policy=TruthlessPolicy(),
    ).run("finish")

    assert state.final_status == "blocked"
    assert state.final_reason == "truthless policy used"


def test_loop_preserves_truthless_confirmer(tmp_path: Path) -> None:
    class TruthlessConfirmer:
        def __bool__(self) -> bool:
            raise RuntimeError("confirmer truthiness unavailable")

        def __call__(
            self,
            decision: PermissionDecision,
            action: ToolCallAction,
        ) -> bool:
            return True

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="write_record",
                    call_id="write-1",
                    arguments={"value": "hello"},
                ),
                FinalAction(text="done"),
            ]
        ),
        registry=ToolRegistry([WriteRecordTool()]),
        logger=MemoryLogger(),
        confirmer=TruthlessConfirmer(),
    ).run("finish")

    assert state.final_status == "completed"
    assert state.observations[0].policy_decision == "confirm"
    assert state.observations[0].result.success is True


def test_loop_preserves_truthless_run_id(tmp_path: Path) -> None:
    class TruthlessRunId(str):
        def __len__(self) -> int:
            raise RuntimeError("run id length unavailable")

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=MemoryLogger(),
        run_id=TruthlessRunId("run-1"),
    ).run("finish")

    assert state.run_id == "run-1"
    assert type(state.run_id) is str


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


def test_loop_logs_unstringable_final_answer_without_raising(
    tmp_path: Path,
) -> None:
    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("answer unavailable")

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [FinalAction(text=UnstringableString("done"))],
            logger,
        ).run("finish")
    except Exception as exc:
        pytest.fail(f"loop raised instead of finishing run: {exc!r}")

    logged_answer = next(
        event[2]["answer"] for event in logger.events if event[0] == "run_finished"
    )
    assert state.final_status == "completed"
    assert logged_answer == "<uninspectable>"


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


def test_loop_rejects_empty_goal_before_model_call(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=UnexpectedLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("")

    assert state.final_status == "invalid_goal"
    assert state.final_reason == "Goal cannot be blank"
    assert state.goal == ""
    assert not any(event[0] == "model_requested" for event in logger.events)


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
            **default_run_start_runtime(),
            "workspace": str(tmp_path),
            "resumed": True,
            "conversation_turns": 1,
            "session": "work",
        },
    )


def test_loop_logs_runtime_settings_in_run_start(tmp_path: Path) -> None:
    logger = MemoryLogger()
    AgentLoop(
        config=RunConfig(
            workspace=tmp_path,
            model="gpt-5.4-mini",
            reasoning_effort="medium",
            text_verbosity="low",
            max_steps=7,
            model_retry_attempts=3,
            permission_profile="trusted",
            confirm_risky_actions=False,
            enabled_tools=["record"],
            tool_modules=["example.tools"],
        ),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("continue")

    payload = logger.events[0][2]
    assert payload["model"] == "gpt-5.4-mini"
    assert payload["reasoning_effort"] == "medium"
    assert payload["text_verbosity"] == "low"
    assert payload["max_steps"] == 7
    assert payload["model_retry_attempts"] == 3
    assert payload["permission_profile"] == "trusted"
    assert payload["confirm_risky_actions"] is False
    assert payload["enabled_tools"] == ["record"]
    assert payload["tool_modules"] == ["example.tools"]


def test_loop_logs_registered_tool_inventory_in_run_start(tmp_path: Path) -> None:
    logger = MemoryLogger()
    AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool(), AskUserTool()]),
        logger=logger,
    ).run("continue")

    payload = logger.events[0][2]
    assert payload["registered_tools"] == [
        {
            "name": "record",
            "description": "Record a value.",
            "risk": "read",
            "args_schema_hash": schema_hash(RecordTool.args_schema),
            "implementation": tool_implementation(RecordTool),
        },
        {
            "name": "ask_user",
            "description": "Ask the user one direct question.",
            "risk": "interactive",
            "args_schema_hash": schema_hash(AskUserTool.args_schema),
            "implementation": tool_implementation(AskUserTool),
        },
    ]


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
            **default_run_start_runtime(),
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
            **default_run_start_runtime(),
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


def test_loop_sanitizes_unstringable_trace_context_keys(tmp_path: Path) -> None:
    class UnstringableKey:
        def __str__(self) -> str:
            raise RuntimeError("key unavailable")

    logger = MemoryLogger()
    trace = MemoryTraceSink()

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
        trace=trace,
    ).run("finish", trace_context={UnstringableKey(): "value"})

    logged_payload = next(
        event[2] for event in logger.events if event[0] == "run_started"
    )
    traced_payload = next(
        event.payload for event in trace.events if event.name == "run_started"
    )

    assert state.final_status == "completed"
    assert logged_payload["<uninspectable>"] == "value"
    assert traced_payload["<uninspectable>"] == "value"


def test_loop_ignores_uninspectable_trace_context_mapping(tmp_path: Path) -> None:
    class ExplodingTraceContext(dict):
        def items(self):  # type: ignore[override]
            raise RuntimeError("trace context unavailable")

    logger = MemoryLogger()

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([FinalAction(text="done")]),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("finish", trace_context=ExplodingTraceContext({"session": "work"}))

    assert state.final_status == "completed"
    assert logger.events[0] == (
        "run_started",
        0,
        {
            "goal": "finish",
            **default_run_start_runtime(),
            "workspace": str(tmp_path),
            "resumed": False,
            "conversation_turns": 0,
        },
    )


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


def test_loop_normalizes_goal_text_subclasses(tmp_path: Path) -> None:
    class StickyString(str):
        def __str__(self) -> str:
            return self

    seen_goals: list[str] = []

    class InspectingLLM:
        def next_action(self, state, registry):
            seen_goals.append(state.goal)
            return FinalAction(text="done")

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=InspectingLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=MemoryLogger(),
    ).run(StickyString("finish"))

    assert state.final_status == "completed"
    assert state.goal == "finish"
    assert type(state.goal) is str
    assert seen_goals == ["finish"]
    assert type(seen_goals[0]) is str


def test_loop_rejects_unstringable_goal_before_model_call(tmp_path: Path) -> None:
    class UnstringableGoal:
        def __str__(self) -> str:
            raise RuntimeError("goal unavailable")

    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=UnexpectedLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run(UnstringableGoal())  # type: ignore[arg-type]

    assert state.final_status == "invalid_goal"
    assert state.final_reason == "Goal could not be inspected"
    assert state.goal == "<uninspectable>"
    assert logger.events[0][2]["goal"] == "<uninspectable>"
    assert not any(event[0] == "model_requested" for event in logger.events)
    assert logger.events[-1] == (
        "run_finished",
        0,
        {
            "status": "invalid_goal",
            "answer": None,
            "reason": "Goal could not be inspected",
        },
    )


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
            "attempt": 2,
            "max_attempts": 2,
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


def test_loop_logs_run_finished_with_lengthless_reason(tmp_path: Path) -> None:
    class LengthlessReason(str):
        def __len__(self) -> int:
            raise RuntimeError("reason length unavailable")

    logger = MemoryLogger()
    loop = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM([]),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    )
    state = RunState(run_id="run-1", workspace=tmp_path, goal="finish")
    state.final_status = "model_error"
    state.final_reason = LengthlessReason("Model call failed")

    loop._log_run_finished(state)  # noqa: SLF001

    run_finished = next(event for event in logger.events if event[0] == "run_finished")
    assert run_finished[2]["reason"] == "Model call failed"


def test_loop_retries_transient_model_error_before_returning_action(
    tmp_path: Path,
) -> None:
    class FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        def next_action(self, state, registry):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary model outage")
            return FinalAction(text="recovered")

    logger = MemoryLogger()
    llm = FlakyLLM()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path, model_retry_attempts=2),
        llm=llm,
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("finish")

    assert state.final_status == "completed"
    assert state.final_answer == "recovered"
    assert state.step_count == 1
    assert llm.calls == 2
    assert (
        "model_retry",
        1,
        {
            "attempt": 1,
            "next_attempt": 2,
            "max_attempts": 2,
            "error_type": "RuntimeError",
            "error": "temporary model outage",
        },
    ) in logger.events
    assert not any(event[0] == "run_error" for event in logger.events)


def test_loop_returns_model_error_after_retry_attempts_are_exhausted(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path, model_retry_attempts=2),
        llm=FailingLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("finish")

    assert state.final_status == "model_error"
    assert state.final_reason == "Model call failed: RuntimeError"
    assert [
        event[0]
        for event in logger.events
        if event[0] in {"model_retry", "run_error"}
    ] == ["model_retry", "run_error"]
    assert (
        "run_error",
        1,
        {
            "status": "model_error",
            "attempt": 2,
            "max_attempts": 2,
            "error_type": "RuntimeError",
            "error": "model unavailable",
        },
    ) in logger.events


def test_loop_logs_unstringable_model_error_and_returns_state(tmp_path: Path) -> None:
    class UnstringableException(Exception):
        def __str__(self) -> str:
            raise RuntimeError("message unavailable")

    class UnstringableLLM:
        def next_action(self, state, registry):
            raise UnstringableException()

    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=UnstringableLLM(),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("finish")

    assert state.final_status == "model_error"
    assert state.final_reason == "Model call failed: UnstringableException"
    assert (
        "run_error",
        1,
        {
            "status": "model_error",
            "attempt": 2,
            "max_attempts": 2,
            "error_type": "UnstringableException",
            "error": "UnstringableException",
        },
    ) in logger.events


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


def test_loop_rejects_oversized_final_status_before_logging(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    oversized_status = "a" * 513

    state = make_loop(
        tmp_path,
        [FinalAction(text="done", status=oversized_status)],
        logger,
    ).run("finish")

    reason = "Model returned invalid final action: status exceeds 512 bytes"
    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
    assert state.final_answer is None
    assert oversized_status not in str(logger.events)
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "invalid_final_action",
            "error": reason,
        },
    ) in logger.events


def test_loop_rejects_unknown_final_action_status(tmp_path: Path) -> None:
    logger = MemoryLogger()

    state = make_loop(
        tmp_path,
        [FinalAction(text="done", status="paused")],
        logger,
    ).run("finish")

    reason = "Model returned invalid final action: unknown status paused"
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


def test_loop_rejects_uninspectable_final_action_text_attribute(
    tmp_path: Path,
) -> None:
    class FinalActionWithExplodingText(FinalAction):
        def __getattribute__(self, name):
            if name == "text":
                raise RuntimeError("final text unavailable")
            return super().__getattribute__(name)

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [FinalActionWithExplodingText(text="done")],
            logger,
        ).run("finish")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid final action: {exc!r}")

    reason = "Model returned invalid final action: text could not be inspected"
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


def test_loop_rejects_uninspectable_final_action_status_attribute(
    tmp_path: Path,
) -> None:
    class FinalActionWithExplodingStatus(FinalAction):
        def __getattribute__(self, name):
            if name == "status":
                raise RuntimeError("final status unavailable")
            return super().__getattribute__(name)

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [FinalActionWithExplodingStatus(text="done")],
            logger,
        ).run("finish")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid final action: {exc!r}")

    reason = "Model returned invalid final action: status could not be inspected"
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


def test_loop_uses_final_action_values_captured_during_validation(
    tmp_path: Path,
) -> None:
    class VolatileFinalAction(FinalAction):
        def __init__(self, text: str, status: str = "completed") -> None:
            super().__init__(text=text, status=status)
            object.__setattr__(self, "_text_reads", 0)
            object.__setattr__(self, "_status_reads", 0)

        def __getattribute__(self, name):
            if name == "text":
                reads = object.__getattribute__(self, "_text_reads")
                object.__setattr__(self, "_text_reads", reads + 1)
                if reads:
                    raise RuntimeError("final text unavailable")
            if name == "status":
                reads = object.__getattribute__(self, "_status_reads")
                object.__setattr__(self, "_status_reads", reads + 1)
                if reads:
                    raise RuntimeError("final status unavailable")
            return super().__getattribute__(name)

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [VolatileFinalAction(text="done", status="completed")],
            logger,
        ).run("finish")
    except Exception as exc:
        pytest.fail(f"loop raised instead of reusing final action values: {exc!r}")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "completed",
            "answer": "done",
        },
    )


def test_loop_normalizes_final_action_text_subclasses(tmp_path: Path) -> None:
    class StickyString(str):
        def __str__(self) -> str:
            return self

        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

    state = make_loop(
        tmp_path,
        [
            FinalAction(
                text=StickyString("done"),
                status=StickyString("completed"),
            )
        ],
    ).run("finish")

    assert state.final_status == "completed"
    assert type(state.final_status) is str
    assert state.final_answer == "done"
    assert type(state.final_answer) is str


def test_loop_rejects_unencodable_final_status_before_logging(
    tmp_path: Path,
) -> None:
    class UnencodableString(str):
        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot encode")

    logger = MemoryLogger()

    state = make_loop(
        tmp_path,
        [FinalAction(text="done", status=UnencodableString("completed"))],
        logger,
    ).run("finish")

    reason = "Model returned invalid final action: status could not be inspected"
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


def test_loop_rejects_uninspectable_final_status_before_logging(
    tmp_path: Path,
) -> None:
    class UninspectableString(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot strip")

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [FinalAction(text="done", status=UninspectableString("completed"))],
            logger,
        ).run("finish")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid final action: {exc!r}")

    reason = "Model returned invalid final action: status could not be inspected"
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


def test_loop_rejects_unencodable_tool_call_metadata(
    tmp_path: Path,
) -> None:
    class UnencodableString(str):
        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot encode")

    cases = [
        (
            ToolCallAction(
                tool_name=UnencodableString("record"),
                arguments={"value": "x"},
                call_id="call-1",
            ),
            "Model returned invalid tool call: tool_name could not be inspected",
        ),
        (
            ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id=UnencodableString("call-1"),
            ),
            "Model returned invalid tool call: call_id could not be inspected",
        ),
    ]
    for action, reason in cases:
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
        assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_uninspectable_tool_call_metadata(
    tmp_path: Path,
) -> None:
    class UninspectableString(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot strip")

    cases = [
        (
            ToolCallAction(
                tool_name=UninspectableString("record"),
                arguments={"value": "x"},
                call_id="call-1",
            ),
            "Model returned invalid tool call: tool_name could not be inspected",
        ),
        (
            ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id=UninspectableString("call-1"),
            ),
            "Model returned invalid tool call: call_id could not be inspected",
        ),
    ]
    for action, reason in cases:
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
        assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_uninspectable_tool_call_arguments_attribute(
    tmp_path: Path,
) -> None:
    class ToolCallWithExplodingArguments(ToolCallAction):
        def __getattribute__(self, name):
            if name == "arguments":
                raise RuntimeError("arguments unavailable")
            return super().__getattribute__(name)

    action = ToolCallWithExplodingArguments("record", {"value": "x"}, "call-1")
    logger = MemoryLogger()

    try:
        state = make_loop(tmp_path, [action], logger).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid action: {exc!r}")

    reason = "Model returned invalid tool call: arguments could not be inspected"
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
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_uninspectable_tool_call_name_attribute(
    tmp_path: Path,
) -> None:
    class ToolCallWithExplodingName(ToolCallAction):
        def __getattribute__(self, name):
            if name == "tool_name":
                raise RuntimeError("tool name unavailable")
            return super().__getattribute__(name)

    action = ToolCallWithExplodingName("record", {"value": "x"}, "call-1")
    logger = MemoryLogger()

    try:
        state = make_loop(tmp_path, [action], logger).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid action: {exc!r}")

    reason = "Model returned invalid tool call: tool_name could not be inspected"
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
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_uninspectable_tool_call_id_attribute(
    tmp_path: Path,
) -> None:
    class ToolCallWithExplodingCallId(ToolCallAction):
        def __getattribute__(self, name):
            if name == "call_id":
                raise RuntimeError("call id unavailable")
            return super().__getattribute__(name)

    action = ToolCallWithExplodingCallId("record", {"value": "x"}, "call-1")
    logger = MemoryLogger()

    try:
        state = make_loop(tmp_path, [action], logger).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid action: {exc!r}")

    reason = "Model returned invalid tool call: call_id could not be inspected"
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
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_ignores_uninspectable_tool_call_provider_metadata(
    tmp_path: Path,
) -> None:
    class ToolCallWithExplodingProviderMetadata(ToolCallAction):
        def __getattribute__(self, name):
            if name == "provider_metadata":
                raise RuntimeError("provider metadata unavailable")
            return super().__getattribute__(name)

    action = ToolCallWithExplodingProviderMetadata(
        "record",
        {"value": "x"},
        "call-1",
    )
    logger = MemoryLogger()

    try:
        state = make_loop(
            tmp_path,
            [action, FinalAction(text="done")],
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of executing tool call: {exc!r}")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert len(state.observations) == 1
    assert state.observations[0].result.success is True
    assert any(event[0] == "tool_started" for event in logger.events)


def test_loop_uses_tool_action_values_captured_during_validation(
    tmp_path: Path,
) -> None:
    class VolatileToolCallAction(ToolCallAction):
        def __init__(
            self,
            tool_name: str,
            arguments: dict[str, object],
            call_id: str,
        ) -> None:
            super().__init__(tool_name, arguments, call_id)
            object.__setattr__(self, "_tool_name_reads", 0)
            object.__setattr__(self, "_call_id_reads", 0)

        def __getattribute__(self, name):
            if name == "tool_name":
                reads = object.__getattribute__(self, "_tool_name_reads")
                object.__setattr__(self, "_tool_name_reads", reads + 1)
                if reads:
                    raise RuntimeError("tool name unavailable")
            if name == "call_id":
                reads = object.__getattribute__(self, "_call_id_reads")
                object.__setattr__(self, "_call_id_reads", reads + 1)
                if reads:
                    raise RuntimeError("call id unavailable")
            return super().__getattribute__(name)

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [
                VolatileToolCallAction("record", {"value": "x"}, "call-1"),
                FinalAction(text="done"),
            ],
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of reusing tool action values: {exc!r}")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert state.observations[0].tool_name == "record"
    assert state.observations[0].call_id == "call-1"
    assert state.observations[0].result.success is True


def test_loop_recovers_when_tool_name_stringification_fails_after_validation(
    tmp_path: Path,
) -> None:
    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("tool name unavailable")

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [
                ToolCallAction(
                    tool_name=UnstringableString("record"),
                    arguments={"value": "x"},
                    call_id="call-1",
                ),
                FinalAction(text="done"),
            ],
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of executing tool call: {exc!r}")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert state.observations[0].result.success is True


def test_loop_normalizes_tool_action_text_subclasses(tmp_path: Path) -> None:
    class StickyString(str):
        def __str__(self) -> str:
            return self

        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self

    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name=StickyString("record"),
                arguments={"value": "x"},
                call_id=StickyString("call-1"),
            ),
            FinalAction(text="done"),
        ],
        logger,
    ).run("record")

    model_action = next(event for event in logger.events if event[0] == "model_action")

    assert state.final_status == "completed"
    assert state.observations[0].tool_name == "record"
    assert type(state.observations[0].tool_name) is str
    assert state.observations[0].call_id == "call-1"
    assert type(state.observations[0].call_id) is str
    assert model_action[2]["tool_name"] == "record"
    assert type(model_action[2]["tool_name"]) is str
    assert model_action[2]["call_id"] == "call-1"
    assert type(model_action[2]["call_id"]) is str


def test_loop_strips_tool_action_metadata_for_execution(tmp_path: Path) -> None:
    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name=" record ",
                arguments={"value": "x"},
                call_id=" call-1 ",
            ),
            FinalAction(text="done"),
        ],
        logger,
    ).run("record")

    model_action = next(event for event in logger.events if event[0] == "model_action")

    assert state.final_status == "completed"
    assert state.final_answer == "done"
    assert state.observations[0].tool_name == "record"
    assert state.observations[0].call_id == "call-1"
    assert state.observations[0].result.success is True
    assert model_action[2]["tool_name"] == "record"
    assert model_action[2]["call_id"] == "call-1"


def test_loop_uses_normalized_tool_name_for_registry_lookup(tmp_path: Path) -> None:
    class UnhashableString(str):
        def __hash__(self) -> int:
            raise RuntimeError("tool name hash unavailable")

    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                tool_name=UnhashableString("record"),
                arguments={"value": "x"},
                call_id="call-1",
            ),
            FinalAction(text="done"),
        ],
        logger,
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].tool_name == "record"
    assert state.observations[0].result.success is True


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


def test_loop_logs_run_snapshot_with_observation_details(tmp_path: Path) -> None:
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

    snapshot = next(event[2] for event in logger.events if event[0] == "run_snapshot")

    assert state.final_status == "completed"
    assert snapshot["goal"] == "record"
    assert snapshot["workspace"] == str(tmp_path)
    assert snapshot["step_count"] == 1
    assert snapshot["sent_observation_count"] == 0
    assert snapshot["repeated_action_count"] == 1
    assert snapshot["observations"] == [
        {
            "call_id": "call-1",
            "tool_name": "record",
            "policy_decision": "allow",
            "result": {
                "success": True,
                "payload": {"value": "x", "workspace": str(tmp_path)},
                "summary": "recorded x",
                "error": None,
            },
        }
    ]


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


def test_loop_executes_iterable_tool_call_batch_without_length(
    tmp_path: Path,
) -> None:
    class ExplodingToolCalls(list):
        def __len__(self) -> int:
            raise RuntimeError("tool call length unavailable")

    state = make_loop(
        tmp_path,
        [
            ToolCallBatchAction(
                tool_calls=ExplodingToolCalls(
                    [
                        ToolCallAction(
                            tool_name="record",
                            arguments={"value": "x"},
                            call_id="call-1",
                        )
                    ]
                )
            ),
            FinalAction(text="done"),
        ],
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].call_id == "call-1"
    assert state.observations[0].result.payload["value"] == "x"


def test_loop_rejects_tool_call_batch_when_iteration_fails(
    tmp_path: Path,
) -> None:
    class PartiallyIterableToolCalls(list):
        def __iter__(self):
            yield ToolCallAction(
                tool_name="record",
                arguments={"value": "x"},
                call_id="call-1",
            )
            raise RuntimeError("tool call batch interrupted")

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [
                ToolCallBatchAction(
                    tool_calls=PartiallyIterableToolCalls(
                        [
                            ToolCallAction(
                                tool_name="record",
                                arguments={"value": "hidden"},
                                call_id="hidden-call",
                            )
                        ]
                    )
                )
            ],
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid batch: {exc!r}")

    reason = "Model returned invalid tool call batch: tool_calls could not be inspected"
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


def test_loop_rejects_unstringable_duplicate_batch_call_id_without_raising(
    tmp_path: Path,
) -> None:
    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("call id unavailable")

    logger = MemoryLogger()
    call_id = UnstringableString("call-1")
    try:
        state = make_loop(
            tmp_path,
            [
                ToolCallBatchAction(
                    tool_calls=[
                        ToolCallAction("record", {"value": "x"}, call_id),
                        ToolCallAction("record", {"value": "y"}, call_id),
                    ]
                )
            ],
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid batch: {exc!r}")

    reason = (
        "Model returned invalid tool call batch: duplicate call_id: <uninspectable>"
    )
    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
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


def test_loop_rejects_tool_call_name_with_invalid_strip_result_without_raising(
    tmp_path: Path,
) -> None:
    class InvalidStripString(str):
        def strip(self, chars=None):  # type: ignore[override]
            return []

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [
                ToolCallAction(
                    InvalidStripString("record"),
                    {"value": "x"},
                    "call-1",
                )
            ],
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid tool call: {exc!r}")

    reason = "Model returned invalid tool call: tool_name could not be inspected"
    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
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
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_unhashable_duplicate_batch_call_id_without_raising(
    tmp_path: Path,
) -> None:
    class UnhashableString(str):
        def __hash__(self) -> int:
            raise RuntimeError("call id hash unavailable")

    logger = MemoryLogger()
    call_id = UnhashableString("call-1")
    try:
        state = make_loop(
            tmp_path,
            [
                ToolCallBatchAction(
                    tool_calls=[
                        ToolCallAction("record", {"value": "x"}, call_id),
                        ToolCallAction("record", {"value": "y"}, call_id),
                    ]
                )
            ],
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording invalid batch: {exc!r}")

    reason = "Model returned invalid tool call batch: duplicate call_id: call-1"
    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
    assert state.observations == []
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_rejects_batch_tool_call_metadata_before_duplicate_check(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    oversized_call_id = "a" * 513

    state = make_loop(
        tmp_path,
        [
            ToolCallBatchAction(
                tool_calls=[
                    ToolCallAction(
                        tool_name="record",
                        arguments={"value": "x"},
                        call_id=oversized_call_id,
                    ),
                    ToolCallAction(
                        tool_name="record",
                        arguments={"value": "y"},
                        call_id=oversized_call_id,
                    ),
                ]
            )
        ],
        logger,
    ).run("record")

    reason = "Model returned invalid tool call: call_id exceeds 512 bytes"
    assert state.final_status == "invalid_action"
    assert state.final_reason == reason
    assert state.final_answer is None
    assert state.observations == []
    assert oversized_call_id not in str(logger.events)
    assert (
        "run_error",
        1,
        {
            "status": "invalid_action",
            "error_type": "invalid_tool_call",
            "error": reason,
        },
    ) in logger.events
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


def test_loop_denies_tool_when_confirmer_raises_unstringable_exception(
    tmp_path: Path,
) -> None:
    class UnstringableException(Exception):
        def __str__(self) -> str:
            raise RuntimeError("message unavailable")

    logger = MemoryLogger()

    def failing_confirmer(decision, action) -> bool:
        raise UnstringableException()

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
    assert state.final_reason == (
        "Permission confirmation failed: UnstringableException"
    )
    observation = state.observations[0]
    assert observation.policy_decision == "confirm"
    assert observation.result.success is False
    assert observation.result.summary == (
        "Permission confirmation failed: UnstringableException"
    )
    assert observation.result.error == "UnstringableException"
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "denied",
            "answer": None,
            "reason": "Permission confirmation failed: UnstringableException",
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


def test_loop_blocks_tool_when_policy_decision_raises(tmp_path: Path) -> None:
    class FailingPolicy(PermissionPolicy):
        def decide(
            self,
            tool_name: str,
            args: dict[str, object],
            risk: str,
        ) -> PermissionDecision:
            raise RuntimeError("policy backend unavailable")

    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
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
        policy=FailingPolicy(),
    ).run("record")

    expected_reason = "Permission policy failed: RuntimeError"
    assert state.final_status == "blocked"
    assert state.final_reason == expected_reason
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "block"
    assert observation.result.success is False
    assert observation.result.summary == expected_reason
    assert observation.result.error == "policy backend unavailable"
    assert not any(event[0] == "tool_started" for event in logger.events)
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "blocked",
            "answer": None,
            "reason": expected_reason,
        },
    )


def test_loop_blocks_tool_when_policy_decision_outcome_raises(
    tmp_path: Path,
) -> None:
    class UninspectableDecision:
        @property
        def outcome(self) -> str:
            raise RuntimeError("decision outcome unavailable")

        @property
        def reason(self) -> str:
            return "unreachable"

    class UninspectableDecisionPolicy(PermissionPolicy):
        def decide(
            self,
            tool_name: str,
            args: dict[str, object],
            risk: str,
        ) -> PermissionDecision:
            return UninspectableDecision()

    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
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
        policy=UninspectableDecisionPolicy(),
    ).run("record")

    expected_reason = "Permission decision invalid: RuntimeError"
    assert state.final_status == "blocked"
    assert state.final_reason == expected_reason
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "block"
    assert observation.result.success is False
    assert observation.result.summary == expected_reason
    assert observation.result.error == "decision outcome unavailable"
    assert not any(event[0] == "tool_started" for event in logger.events)
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "blocked",
            "answer": None,
            "reason": expected_reason,
        },
    )


def test_loop_blocks_tool_when_policy_decision_outcome_is_unknown(
    tmp_path: Path,
) -> None:
    class UnknownOutcomePolicy(PermissionPolicy):
        def decide(
            self,
            tool_name: str,
            args: dict[str, object],
            risk: str,
        ) -> PermissionDecision:
            return PermissionDecision("approve", "unknown policy outcome")

    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
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
        policy=UnknownOutcomePolicy(),
    ).run("record")

    expected_reason = "Permission decision invalid: unknown outcome approve"
    assert state.final_status == "blocked"
    assert state.final_reason == expected_reason
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "block"
    assert observation.result.success is False
    assert observation.result.summary == expected_reason
    assert observation.result.error == "Invalid permission decision"
    assert not any(event[0] == "tool_started" for event in logger.events)
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "blocked",
            "answer": None,
            "reason": expected_reason,
        },
    )


def test_loop_blocks_tool_when_policy_decision_reason_is_invalid(
    tmp_path: Path,
) -> None:
    class InvalidReasonDecisionPolicy(PermissionPolicy):
        def decide(
            self,
            tool_name: str,
            args: dict[str, object],
            risk: str,
        ) -> PermissionDecision:
            return PermissionDecision("block", None)  # type: ignore[arg-type]

    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
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
        policy=InvalidReasonDecisionPolicy(),
    ).run("record")

    expected_reason = (
        "Permission decision invalid: reason must be a non-empty string"
    )
    assert state.final_status == "blocked"
    assert state.final_reason == expected_reason
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "block"
    assert observation.result.success is False
    assert observation.result.summary == expected_reason
    assert observation.result.error == "Invalid permission decision"
    assert not any(event[0] == "tool_started" for event in logger.events)
    assert logger.events[-1] == (
        "run_finished",
        1,
        {
            "status": "blocked",
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


def test_loop_stops_with_unknown_tool_when_name_stringification_fails(
    tmp_path: Path,
) -> None:
    class UnstringableString(str):
        def __str__(self) -> str:
            raise RuntimeError("tool name unavailable")

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [
                ToolCallAction(
                    tool_name=UnstringableString("missing"),
                    arguments={"value": "x"},
                    call_id="call-1",
                )
            ],
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording unknown tool: {exc!r}")

    assert state.final_status == "unknown_tool"
    assert state.final_reason == "Unknown tool: <uninspectable>"
    assert state.observations[0].result.summary == "Unknown tool: <uninspectable>"


def test_loop_uses_captured_tool_action_values_for_unknown_tools(
    tmp_path: Path,
) -> None:
    class VolatileUnknownToolCall(ToolCallAction):
        def __init__(
            self,
            tool_name: str,
            arguments: dict[str, object],
            call_id: str,
        ) -> None:
            super().__init__(tool_name, arguments, call_id)
            object.__setattr__(self, "_tool_name_reads", 0)
            object.__setattr__(self, "_arguments_reads", 0)

        def __getattribute__(self, name):
            if name == "tool_name":
                reads = object.__getattribute__(self, "_tool_name_reads")
                object.__setattr__(self, "_tool_name_reads", reads + 1)
                if reads:
                    raise RuntimeError("tool name unavailable")
            if name == "arguments":
                reads = object.__getattribute__(self, "_arguments_reads")
                object.__setattr__(self, "_arguments_reads", reads + 1)
                if reads:
                    raise RuntimeError("arguments unavailable")
            return super().__getattribute__(name)

    logger = MemoryLogger()
    try:
        state = make_loop(
            tmp_path,
            [VolatileUnknownToolCall("missing", {"value": "x"}, "call-1")],
            logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised instead of recording unknown tool: {exc!r}")

    assert state.final_status == "unknown_tool"
    assert state.final_reason == "Unknown tool: missing"
    assert state.observations[0].tool_name == "missing"
    assert state.observations[0].call_id == "call-1"
    assert state.observations[0].result.payload == {
        "tool_name": "missing",
        "arguments": {"value": "x"},
    }


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


def test_loop_bounds_large_validation_error_lists(tmp_path: Path) -> None:
    logger = MemoryLogger()
    arguments = {
        "value": "ok",
        **{f"extra_{index}": "x" for index in range(200)},
    }

    state = make_loop(
        tmp_path,
        [
            ToolCallAction("record", arguments, "call-1"),
            FinalAction(text="recovered"),
        ],
        logger,
    ).run("record")

    validation_errors = state.observations[0].result.payload["validation_errors"]
    assert isinstance(validation_errors, list)
    assert len(validation_errors) == 50
    assert validation_errors[0] == "Unexpected argument: extra_0"
    assert validation_errors[-1] == "[truncated 151 validation errors]"
    assert "extra_199" not in str(validation_errors)
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


def test_loop_rejects_nested_non_string_tool_argument_names_before_execution(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    loop = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction("object_record", {"payload": {1: "x"}}, "call-1"),
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
        "Tool argument names must be strings"
    ]
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_recovers_when_tool_argument_inspection_fails(
    tmp_path: Path,
) -> None:
    class ExplodingValuesArguments(dict):
        def values(self):  # type: ignore[override]
            raise RuntimeError("argument values unavailable")

    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                "record",
                ExplodingValuesArguments({"value": "ok"}),
                "call-1",
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
    assert observation.result.error == "Invalid arguments"
    assert observation.result.payload["validation_errors"] == [
        "Tool arguments could not be inspected"
    ]
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_recovers_when_logging_tool_arguments_fails(
    tmp_path: Path,
) -> None:
    class ExplodingItemsArguments(dict):
        def items(self):  # type: ignore[override]
            raise RuntimeError("argument items unavailable")

    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                "record",
                ExplodingItemsArguments({"value": "ok"}),
                "call-1",
            ),
            FinalAction(text="recovered"),
        ],
        logger,
    ).run("record")

    model_action = next(event for event in logger.events if event[0] == "model_action")
    observation = state.observations[0]

    assert state.final_status == "completed"
    assert state.final_answer == "recovered"
    assert model_action[2]["arguments"] == "<uninspectable>"
    assert observation.result.error == "Invalid arguments"
    assert observation.result.payload["arguments"] == "<uninspectable>"
    assert observation.result.payload["validation_errors"] == [
        "Tool arguments could not be inspected"
    ]
    assert not any(event[0] == "tool_started" for event in logger.events)


def test_loop_preserves_unstringable_logged_argument_entries(
    tmp_path: Path,
) -> None:
    class UnstringableKey:
        def __str__(self) -> str:
            raise RuntimeError("key unavailable")

    class UnstringableValue:
        def __str__(self) -> str:
            raise RuntimeError("value unavailable")

    logger = MemoryLogger()
    state = make_loop(
        tmp_path,
        [
            ToolCallAction(
                "record",
                {UnstringableKey(): UnstringableValue()},  # type: ignore[dict-item]
                "call-1",
            ),
            FinalAction(text="recovered"),
        ],
        logger,
    ).run("record")

    model_action = next(event for event in logger.events if event[0] == "model_action")
    observation = state.observations[0]

    assert state.final_status == "completed"
    assert state.final_answer == "recovered"
    assert model_action[2]["arguments"] == {
        "<uninspectable>": "<uninspectable>"
    }
    assert observation.result.error == "Invalid arguments"
    assert observation.result.payload["arguments"] == {
        "<uninspectable>": "<uninspectable>"
    }
    assert observation.result.payload["validation_errors"] == [
        "Tool argument names must be strings"
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
    assert observation.result.payload["tool_name"] == "record"
    assert observation.result.payload["arguments"] == {str(bad_key): "x"}
    assert observation.result.payload["validation_errors"] == [
        "Tool argument names must be strings"
    ]
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


def test_loop_converts_unstringable_tool_exception_to_observation(
    tmp_path: Path,
) -> None:
    class UnstringableException(Exception):
        def __str__(self) -> str:
            raise RuntimeError("message unavailable")

    class UnstringableErrorTool(RecordTool):
        name = "unstringable_error"

        def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
            raise UnstringableException()

    logger = MemoryLogger()
    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="unstringable_error",
                    arguments={"value": "x"},
                    call_id="call-1",
                )
            ]
        ),
        registry=ToolRegistry([UnstringableErrorTool()]),
        logger=logger,
    ).run("record")

    assert state.final_status == "tool_error"
    observation = state.observations[0]
    assert observation.result.summary == (
        "Tool raised an exception: UnstringableException"
    )
    assert observation.result.error == "UnstringableException"
    assert (
        "tool_finished",
        1,
        {
            "tool_name": "unstringable_error",
            "success": False,
            "summary": "Tool raised an exception: UnstringableException",
            "error": "UnstringableException",
        },
    ) in logger.events


def test_loop_bounds_stored_and_logged_tool_error_message(tmp_path: Path) -> None:
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
    stored_error = state.observations[0].result.error
    assert stored_error is not None
    assert stored_error.startswith("x" * 40)
    assert "[truncated" in stored_error
    assert len(stored_error) < 5_000
    assert isinstance(logged_error, dict)
    assert logged_error["truncated"] is True
    assert logged_error["bytes"] == len(stored_error.encode("utf-8"))
    assert str(logged_error["preview"]).startswith("xxxxxxxxxxxxxxxx")
    assert len(str(logged_error["preview"])) < 300
    assert traced_error == logged_error
    assert error_message not in str(state.observations)
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


def test_loop_converts_uninspectable_tool_result_to_observation(
    tmp_path: Path,
) -> None:
    logger = MemoryLogger()
    try:
        state = AgentLoop(
            config=RunConfig(workspace=tmp_path),
            llm=ScriptedLLM(
                [
                    ToolCallAction(
                        tool_name="uninspectable_result",
                        arguments={"value": "x"},
                        call_id="call-1",
                    )
                ]
            ),
            registry=ToolRegistry([UninspectableResultTool()]),
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
        "Tool returned malformed result: success could not be inspected"
    )
    assert observation.result.error == "Malformed tool result"
    assert observation.result.payload == {
        "tool_name": "uninspectable_result",
        "arguments": {"value": "x"},
        "validation_errors": ["success could not be inspected"],
    }
    assert (
        "tool_finished",
        1,
        {
            "tool_name": "uninspectable_result",
            "success": False,
            "summary": (
                "Tool returned malformed result: success could not be inspected"
            ),
            "error": "Malformed tool result",
        },
    ) in logger.events


def test_loop_normalizes_tool_result_before_recording(tmp_path: Path) -> None:
    logger = MemoryLogger()
    try:
        state = AgentLoop(
            config=RunConfig(workspace=tmp_path),
            llm=ScriptedLLM(
                [
                    ToolCallAction(
                        tool_name="flaky_result",
                        arguments={"value": "x"},
                        call_id="call-1",
                    ),
                    FinalAction(text="done"),
                ]
            ),
            registry=ToolRegistry([FlakyResultTool()]),
            logger=logger,
        ).run("record")
    except Exception as exc:
        pytest.fail(f"loop raised after validating a tool result: {exc!r}")

    assert state.final_status == "completed"
    assert len(state.observations) == 1
    observation = state.observations[0]
    assert observation.policy_decision == "allow"
    assert observation.result.success is True
    assert observation.result.payload == {"value": "x"}
    assert observation.result.summary == "flaky ok"
    assert (
        "tool_finished",
        1,
        {
            "tool_name": "flaky_result",
            "success": True,
            "summary": "flaky ok",
            "error": None,
        },
    ) in logger.events


def test_loop_bounds_stored_successful_tool_result_payload(tmp_path: Path) -> None:
    large_payload = "x" * 1_100_000

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="large_payload",
                    arguments={"value": "x"},
                    call_id="call-1",
                ),
                FinalAction(text="done"),
            ]
        ),
        registry=ToolRegistry([LargePayloadTool()]),
        logger=MemoryLogger(),
    ).run("record")

    payload = state.observations[0].result.payload
    assert state.final_status == "completed"
    assert payload["truncated"] is True
    assert payload["bytes"] > 1_048_576
    assert str(payload["preview"]).startswith('{"content":"xxxxxxxxxxxxxxxx')
    assert large_payload not in str(state.observations)


def test_loop_preserves_unstringable_tool_result_payload_entries(
    tmp_path: Path,
) -> None:
    class UnstringableKey:
        def __str__(self) -> str:
            raise RuntimeError("key unavailable")

    class UnstringableValue:
        def __str__(self) -> str:
            raise RuntimeError("value unavailable")

    class UnstringablePayloadTool(RecordTool):
        name = "unstringable_payload"

        def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
            return ToolResult.model_construct(
                success=True,
                payload={UnstringableKey(): UnstringableValue()},
                summary="ok",
                error=None,
            )

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="unstringable_payload",
                    arguments={"value": "x"},
                    call_id="call-1",
                ),
                FinalAction(text="done"),
            ]
        ),
        registry=ToolRegistry([UnstringablePayloadTool()]),
        logger=MemoryLogger(),
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].result.payload == {
        "<uninspectable>": "<uninspectable>"
    }


def test_loop_preserves_iterable_tool_result_payload_without_length(
    tmp_path: Path,
) -> None:
    class ExplodingPayloadItems(list):
        def __len__(self) -> int:
            raise RuntimeError("payload length unavailable")

    class LengthlessPayloadTool(RecordTool):
        name = "lengthless_payload"

        def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
            return ToolResult.model_construct(
                success=True,
                payload={
                    "items": ExplodingPayloadItems(
                        [{"name": "first"}, {"name": "second"}]
                    )
                },
                summary="ok",
                error=None,
            )

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="lengthless_payload",
                    arguments={"value": "x"},
                    call_id="call-1",
                ),
                FinalAction(text="done"),
            ]
        ),
        registry=ToolRegistry([LengthlessPayloadTool()]),
        logger=MemoryLogger(),
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].result.payload["items"] == [
        {"name": "first"},
        {"name": "second"},
    ]


def test_loop_bounds_stored_successful_tool_result_summary(tmp_path: Path) -> None:
    large_summary = "s" * 1_100_000

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="large_summary",
                    arguments={"value": "x"},
                    call_id="call-1",
                ),
                FinalAction(text="done"),
            ]
        ),
        registry=ToolRegistry([LargeSummaryTool()]),
        logger=MemoryLogger(),
    ).run("record")

    summary = state.observations[0].result.summary
    assert state.final_status == "completed"
    assert summary.startswith("s" * 40)
    assert "[truncated" in summary
    assert len(summary) < 5_000
    assert large_summary not in str(state.observations)


def test_loop_accepts_stored_tool_result_summary_without_length(
    tmp_path: Path,
) -> None:
    class LengthlessSummary(str):
        def __len__(self) -> int:
            raise RuntimeError("summary length unavailable")

    class LengthlessSummaryTool(RecordTool):
        name = "lengthless_summary"

        def execute(self, args: dict[str, object], context: ToolContext) -> ToolResult:
            return ToolResult.model_construct(
                success=True,
                payload={"ok": True},
                summary=LengthlessSummary("lengthless ok"),
            )

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="lengthless_summary",
                    arguments={"value": "x"},
                    call_id="call-1",
                ),
                FinalAction(text="done"),
            ]
        ),
        registry=ToolRegistry([LengthlessSummaryTool()]),
        logger=MemoryLogger(),
    ).run("record")

    assert state.final_status == "completed"
    assert state.observations[0].result.summary == "lengthless ok"


def test_loop_bounds_stored_unknown_tool_arguments(tmp_path: Path) -> None:
    large_argument = "x" * 1_100_000

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="missing_tool",
                    arguments={"value": large_argument},
                    call_id="call-1",
                )
            ]
        ),
        registry=ToolRegistry([RecordTool()]),
        logger=MemoryLogger(),
    ).run("record")

    payload = state.observations[0].result.payload
    assert state.final_status == "unknown_tool"
    assert payload["truncated"] is True
    assert payload["bytes"] > 1_048_576
    assert str(payload["preview"]).startswith('{"tool_name":"missing_tool"')
    assert large_argument not in str(state.observations)


def test_loop_bounds_wide_tool_arguments_before_storing(
    tmp_path: Path,
) -> None:
    values = list(range(210))
    mapping = {f"k{index}": index for index in range(210)}
    logger = MemoryLogger()

    state = AgentLoop(
        config=RunConfig(workspace=tmp_path),
        llm=ScriptedLLM(
            [
                ToolCallAction(
                    tool_name="missing_tool",
                    arguments={"values": values, "mapping": mapping},
                    call_id="call-1",
                )
            ]
        ),
        registry=ToolRegistry([RecordTool()]),
        logger=logger,
    ).run("record")

    logged_arguments = next(
        event[2]["arguments"]
        for event in logger.events
        if event[0] == "model_action"
    )
    stored_arguments = state.observations[0].result.payload["arguments"]
    marker = {"truncated": True, "items": 210, "omitted": 11}

    assert state.final_status == "unknown_tool"
    assert logged_arguments["values"][-1] == marker
    assert len(logged_arguments["values"]) == 200
    assert logged_arguments["mapping"]["__truncated_items__"] == marker
    assert len(logged_arguments["mapping"]) == 200
    assert stored_arguments == logged_arguments
    assert 209 not in logged_arguments["values"]
    assert "k209" not in logged_arguments["mapping"]


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


def test_loop_normalizes_unstringable_scalar_conversation_entry(
    tmp_path: Path,
) -> None:
    class UnstringableEntry:
        def __str__(self) -> str:
            raise RuntimeError("entry unavailable")

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
        conversation=UnstringableEntry(),  # type: ignore[arg-type]
    )

    assert state.final_status == "completed"
    assert seen_conversation == [[("user", "<uninspectable>", {})]]
    assert logger.events[0][2]["resumed"] is True
    assert logger.events[0][2]["conversation_turns"] == 1


def test_loop_normalizes_uniterable_conversation_history(tmp_path: Path) -> None:
    class UniterableConversation(list):
        def __iter__(self):
            raise RuntimeError("conversation iterator unavailable")

        def __str__(self) -> str:
            raise RuntimeError("conversation unavailable")

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
            conversation=UniterableConversation(["hidden"]),
        )
    except Exception as exc:
        pytest.fail(f"loop raised instead of normalizing conversation: {exc!r}")

    assert state.final_status == "completed"
    assert seen_conversation == [[("user", "<uninspectable>", {})]]
    assert logger.events[0][2]["resumed"] is True
    assert logger.events[0][2]["conversation_turns"] == 1


def test_loop_preserves_conversation_entries_before_iterator_failure(
    tmp_path: Path,
) -> None:
    class PartiallyIterableConversation(list):
        def __iter__(self):
            yield {"role": "user", "content": "before"}
            raise RuntimeError("conversation interrupted")

        def __str__(self) -> str:
            raise RuntimeError("conversation unavailable")

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
            conversation=PartiallyIterableConversation(["hidden"]),
        )
    except Exception as exc:
        pytest.fail(f"loop raised instead of preserving conversation: {exc!r}")

    assert state.final_status == "completed"
    assert seen_conversation == [
        [
            ("user", "before", {}),
            ("user", "<uninspectable>", {}),
        ]
    ]
    assert logger.events[0][2]["conversation_turns"] == 2


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


def test_loop_isolates_uninspectable_mapping_conversation_entry(
    tmp_path: Path,
) -> None:
    class ExplodingMapping(dict):
        def get(self, key, default=None):  # type: ignore[override]
            raise RuntimeError("entry get unavailable")

        def __str__(self) -> str:
            raise RuntimeError("entry unavailable")

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
        conversation=[
            {"role": "user", "content": "before"},
            ExplodingMapping({"role": "assistant", "content": "hidden"}),
            {"role": "user", "content": "after"},
        ],
    )

    assert state.final_status == "completed"
    assert seen_conversation == [
        [
            ("user", "before", {}),
            ("user", "<uninspectable>", {}),
            ("user", "after", {}),
        ]
    ]
    assert logger.events[0][2]["conversation_turns"] == 3


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


def test_loop_sanitizes_unstringable_conversation_message_fields(
    tmp_path: Path,
) -> None:
    class UnstringableValue:
        def __str__(self) -> str:
            raise RuntimeError("field unavailable")

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
            {
                "role": UnstringableValue(),
                "content": UnstringableValue(),
                "metadata": ["bad"],
            }
        ],
    )

    assert state.final_status == "completed"
    assert seen_conversation == [[("<uninspectable>", "<uninspectable>", {})]]


def test_loop_sanitizes_uninspectable_conversation_metadata(
    tmp_path: Path,
) -> None:
    class ExplodingMetadata(dict):
        def __len__(self) -> int:
            raise RuntimeError("metadata length unavailable")

        def items(self):  # type: ignore[override]
            raise RuntimeError("metadata items unavailable")

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
            {
                "role": "user",
                "content": "remember",
                "metadata": ExplodingMetadata({"api_key": "sk-secret123"}),
            }
        ],
    )

    assert state.final_status == "completed"
    assert seen_conversation == [[("user", "remember", {})]]
