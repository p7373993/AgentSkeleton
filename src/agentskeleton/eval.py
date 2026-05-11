from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from agentskeleton.config import RunConfig
from agentskeleton.core.actions import AgentAction, FinalAction, ToolCallAction
from agentskeleton.core.loop import AgentLoop
from agentskeleton.core.state import RunState
from agentskeleton.core.trace import NullTraceSink
from agentskeleton.logging.run_logger import RunLogger
from agentskeleton.tools.registry import ToolRegistry


@dataclass(frozen=True)
class Scenario:
    name: str
    goal: str
    actions: list[AgentAction]
    expect: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    passed: bool
    status: str | None
    answer: str | None
    observations: int
    failures: list[str]
    log_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "passed": self.passed,
            "status": self.status,
            "answer": self.answer,
            "observations": self.observations,
            "failures": self.failures,
            "log": str(self.log_path),
        }


class ScriptedScenarioLLM:
    def __init__(self, actions: list[AgentAction]) -> None:
        self._actions = list(actions)

    def next_action(self, state: RunState, registry: ToolRegistry) -> AgentAction:
        if not self._actions:
            raise RuntimeError("Scenario actions exhausted")
        return self._actions.pop(0)


def load_scenario(path: Path) -> Scenario:
    if not path.exists():
        raise ValueError(f"Scenario file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario file must contain a mapping: {path}")

    goal = raw.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("Scenario must define a non-empty goal")

    raw_actions = raw.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("Scenario must define at least one action")

    expect = raw.get("expect", {})
    if not isinstance(expect, dict):
        raise ValueError("Scenario expect must be a mapping")
    if not expect:
        raise ValueError("Scenario must define expectations")

    name = raw.get("name") or path.stem
    return Scenario(
        name=str(name),
        goal=goal,
        actions=[_parse_action(item, index) for index, item in enumerate(raw_actions)],
        expect=expect,
    )


def run_scenario(
    scenario: Scenario,
    config: RunConfig,
    registry: ToolRegistry,
) -> ScenarioResult:
    run_id = f"eval-{uuid4()}"
    logger = RunLogger(config.logs_dir, run_id)
    loop = AgentLoop(
        config=config,
        llm=ScriptedScenarioLLM(scenario.actions),
        registry=registry,
        logger=logger,
        confirmer=lambda _decision, _action: False,
        run_id=run_id,
        trace=NullTraceSink(),
    )
    state = loop.run(scenario.goal, trace_context={"scenario": scenario.name})
    failures = _compare_expectations(scenario.expect, state)
    return ScenarioResult(
        scenario=scenario.name,
        passed=not failures,
        status=state.final_status,
        answer=state.final_answer,
        observations=len(state.observations),
        failures=failures,
        log_path=Path(logger.path).resolve(),
    )


def _parse_action(raw: object, index: int) -> AgentAction:
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario action {index + 1} must be a mapping")

    action_type = raw.get("type")
    if action_type == "final":
        text = raw.get("text", "")
        status = raw.get("status", "completed")
        return FinalAction(text=str(text), status=str(status))

    if action_type == "tool":
        tool_name = raw.get("tool") or raw.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError(f"Scenario action {index + 1} must define tool")
        arguments = raw.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(f"Scenario action {index + 1} arguments must be a mapping")
        call_id = raw.get("call_id") or f"scenario-call-{index + 1}"
        return ToolCallAction(
            tool_name=tool_name,
            arguments=dict(arguments),
            call_id=str(call_id),
        )

    raise ValueError(f"Unknown scenario action type: {action_type}")


def _compare_expectations(expect: dict[str, object], state: RunState) -> list[str]:
    failures: list[str] = []
    _expect_equal(failures, "status", expect, state.final_status)
    _expect_equal(failures, "answer", expect, state.final_answer)
    if "observations" in expect and expect["observations"] != len(state.observations):
        failures.append(
            "observations expected "
            f"{expect['observations']!r} but got {len(state.observations)!r}"
        )
    return failures


def _expect_equal(
    failures: list[str],
    key: str,
    expect: dict[str, Any],
    actual: object,
) -> None:
    if key in expect and expect[key] != actual:
        failures.append(f"{key} expected {expect[key]!r} but got {actual!r}")
