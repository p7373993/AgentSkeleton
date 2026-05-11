from __future__ import annotations

from collections.abc import Callable
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
from agentskeleton.policy.paths import PathSecurityError, resolve_workspace_path
from agentskeleton.tools.registry import ToolRegistry

RegistryFactory = Callable[[RunConfig], ToolRegistry]


@dataclass(frozen=True)
class Scenario:
    name: str
    domain: str
    goal: str
    actions: list[AgentAction]
    expect: dict[str, object] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    config_overrides: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    domain: str
    passed: bool
    status: str | None
    answer: str | None
    observations: int
    failures: list[str]
    log_path: Path
    workspace: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "domain": self.domain,
            "passed": self.passed,
            "status": self.status,
            "answer": self.answer,
            "observations": self.observations,
            "failures": self.failures,
            "log": str(self.log_path),
            "workspace": str(self.workspace),
        }


@dataclass(frozen=True)
class ScenarioSuiteResult:
    results: list[ScenarioResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        return self.total - self.passed_count

    @property
    def passed(self) -> bool:
        return self.failed_count == 0

    @property
    def domains(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for result in self.results:
            domain = summary.setdefault(
                result.domain,
                {"passed": 0, "failed": 0, "total": 0},
            )
            domain["total"] += 1
            if result.passed:
                domain["passed"] += 1
            else:
                domain["failed"] += 1
        return dict(sorted(summary.items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "total": self.total,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "domains": self.domains,
            "results": [result.to_dict() for result in self.results],
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

    files = _parse_files(raw.get("files", {}))
    config_overrides = _parse_config_overrides(raw.get("config", {}))
    name = raw.get("name") or path.stem
    return Scenario(
        name=str(name),
        domain=str(raw.get("domain") or "general"),
        goal=goal,
        actions=[_parse_action(item, index) for index, item in enumerate(raw_actions)],
        expect=expect,
        files=files,
        config_overrides=config_overrides,
    )


def load_scenario_suite(path: Path) -> list[Scenario]:
    scenario_paths = _scenario_paths(path)
    if not scenario_paths:
        raise ValueError(f"No scenario files found: {path}")
    return [load_scenario(scenario_path) for scenario_path in scenario_paths]


def run_scenario(
    scenario: Scenario,
    config: RunConfig,
    registry: ToolRegistry | None = None,
    registry_factory: RegistryFactory | None = None,
) -> ScenarioResult:
    run_id = f"eval-{uuid4()}"
    scenario_config = _apply_config_overrides(config, scenario.config_overrides)
    scenario_registry = _resolve_registry(
        scenario_config,
        registry,
        registry_factory,
    )
    workspace = _prepare_workspace(scenario_config, run_id, scenario)
    run_config = scenario_config.model_copy(update={"workspace": workspace})
    logger = RunLogger(run_config.logs_dir, run_id)
    loop = AgentLoop(
        config=run_config,
        llm=ScriptedScenarioLLM(scenario.actions),
        registry=scenario_registry,
        logger=logger,
        confirmer=lambda _decision, _action: False,
        run_id=run_id,
        trace=NullTraceSink(),
    )
    state = loop.run(scenario.goal, trace_context={"scenario": scenario.name})
    failures = _compare_expectations(scenario.expect, state, workspace)
    return ScenarioResult(
        scenario=scenario.name,
        domain=scenario.domain,
        passed=not failures,
        status=state.final_status,
        answer=state.final_answer,
        observations=len(state.observations),
        failures=failures,
        log_path=Path(logger.path).resolve(),
        workspace=workspace,
    )


def run_scenario_suite(
    scenarios: list[Scenario],
    config: RunConfig,
    registry: ToolRegistry | None = None,
    registry_factory: RegistryFactory | None = None,
) -> ScenarioSuiteResult:
    return ScenarioSuiteResult(
        [
            run_scenario(
                scenario,
                config,
                registry,
                registry_factory=registry_factory,
            )
            for scenario in scenarios
        ]
    )


def _scenario_paths(path: Path) -> list[Path]:
    if not path.exists():
        raise ValueError(f"Scenario path not found: {path}")
    if path.is_file():
        return [path]
    return sorted(
        [
            scenario_path
            for pattern in ("*.yaml", "*.yml")
            for scenario_path in path.rglob(pattern)
        ],
        key=lambda item: item.as_posix(),
    )


def _parse_files(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("Scenario files must be a mapping")
    return {str(path): str(content) for path, content in raw.items()}


def _parse_config_overrides(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError("Scenario config must be a mapping")
    return dict(raw)


def _apply_config_overrides(
    config: RunConfig,
    overrides: dict[str, object],
) -> RunConfig:
    if not overrides:
        return config
    data = config.model_dump()
    data.update(overrides)
    return RunConfig(**data)


def _resolve_registry(
    config: RunConfig,
    registry: ToolRegistry | None,
    registry_factory: RegistryFactory | None,
) -> ToolRegistry:
    if registry_factory is not None:
        return registry_factory(config)
    if registry is None:
        raise ValueError("Scenario requires a registry or registry factory")
    return registry


def _prepare_workspace(config: RunConfig, run_id: str, scenario: Scenario) -> Path:
    if not scenario.files:
        return config.workspace.resolve()

    workspace = (config.logs_dir / "workspaces" / run_id).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    for requested_path, content in scenario.files.items():
        try:
            target = resolve_workspace_path(workspace, requested_path)
        except PathSecurityError as exc:
            raise ValueError(str(exc)) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return workspace


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


def _compare_expectations(
    expect: dict[str, object],
    state: RunState,
    workspace: Path,
) -> list[str]:
    failures: list[str] = []
    _expect_equal(failures, "status", expect, state.final_status)
    _expect_equal(failures, "answer", expect, state.final_answer)
    if "observations" in expect and expect["observations"] != len(state.observations):
        failures.append(
            "observations expected "
            f"{expect['observations']!r} but got {len(state.observations)!r}"
        )
    _expect_files(failures, expect.get("files"), workspace)
    return failures


def _expect_equal(
    failures: list[str],
    key: str,
    expect: dict[str, Any],
    actual: object,
) -> None:
    if key in expect and expect[key] != actual:
        failures.append(f"{key} expected {expect[key]!r} but got {actual!r}")


def _expect_files(
    failures: list[str],
    raw_files: object,
    workspace: Path,
) -> None:
    if raw_files is None:
        return
    if not isinstance(raw_files, dict):
        failures.append("files expectation must be a mapping")
        return

    for requested_path, expected_content in raw_files.items():
        path_text = str(requested_path)
        try:
            target = resolve_workspace_path(workspace, path_text)
        except PathSecurityError as exc:
            failures.append(str(exc))
            continue
        if not target.exists():
            failures.append(f"file {path_text} expected but was missing")
            continue
        if not target.is_file():
            failures.append(f"file {path_text} expected but was not a file")
            continue
        actual_content = target.read_text(encoding="utf-8")
        expected_text = str(expected_content)
        if actual_content != expected_text:
            failures.append(
                f"file {path_text} expected {expected_text!r} "
                f"but got {actual_content!r}"
            )
