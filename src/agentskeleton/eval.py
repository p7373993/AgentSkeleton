from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from agentskeleton.config import RunConfig
from agentskeleton.core.actions import (
    AgentAction,
    FinalAction,
    ToolCallAction,
    ToolCallBatchAction,
)
from agentskeleton.core.loop import AgentLoop
from agentskeleton.core.state import ConversationMessage, RunState
from agentskeleton.core.trace import NullTraceSink
from agentskeleton.logging.run_logger import RunLogger
from agentskeleton.policy.paths import PathSecurityError, resolve_workspace_path
from agentskeleton.tools.registry import ToolRegistry

RegistryFactory = Callable[[RunConfig], ToolRegistry]
SUITE_MANIFEST_NAMES = {"suite.yaml", "suite.yml"}
MAX_SCENARIO_FILE_BYTES = 2_097_152
MAX_SCENARIO_WORKSPACE_FILES = 128
MAX_SCENARIO_SUITE_FILES = 512
MAX_SCENARIO_ACTIONS = 200
MAX_SCENARIO_BATCH_CALLS = 20
MAX_SCENARIO_USER_ANSWERS = 200
MAX_SCENARIO_CONVERSATION_TURNS = 200
UNINSPECTABLE_VALUE = "<uninspectable>"


@dataclass(frozen=True)
class ScenarioSuiteConfig:
    required_domains: list[str] = field(default_factory=list)
    min_scenarios_per_required_domain: int = 0


@dataclass(frozen=True)
class ScriptedModelError:
    message: str


@dataclass(frozen=True)
class ScriptedInvalidAction:
    type_name: str = "unexpected"


@dataclass(frozen=True)
class ScriptedConversationAssertion:
    contents: list[str]


ScenarioAction = (
    AgentAction
    | ScriptedModelError
    | ScriptedInvalidAction
    | ScriptedConversationAssertion
)


@dataclass(frozen=True)
class Scenario:
    name: str
    domain: str
    goal: str
    actions: list[ScenarioAction]
    expect: dict[str, object] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    config_overrides: dict[str, object] = field(default_factory=dict)
    user_answers: list[str] = field(default_factory=list)
    conversation: list[ConversationMessage] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    domain: str
    run_id: str
    passed: bool
    status: str | None
    answer: str | None
    reason: str | None
    observations: int
    failures: list[str]
    log_path: Path
    workspace: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "domain": self.domain,
            "run_id": self.run_id,
            "passed": self.passed,
            "status": self.status,
            "answer": self.answer,
            "reason": self.reason,
            "observations": self.observations,
            "failures": self.failures,
            "log": str(self.log_path),
            "workspace": str(self.workspace),
        }


@dataclass(frozen=True)
class ScenarioSuiteResult:
    results: list[ScenarioResult]
    required_domains: list[str] = field(default_factory=list)
    min_scenarios_per_required_domain: int = 0

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
        return self.failed_count == 0 and not self.coverage_failures

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

    @property
    def coverage_failures(self) -> list[str]:
        failures: list[str] = []
        domains = self.domains
        for domain in self.required_domains:
            stats = domains.get(domain)
            if stats is None:
                failures.append(f"required domain {domain} has no scenarios")
            elif stats["failed"] > 0:
                failures.append(f"required domain {domain} has failing scenarios")
            if stats is not None and (
                stats["total"] < self.min_scenarios_per_required_domain
            ):
                failures.append(
                    f"required domain {domain} has "
                    f"{_scenario_count_text(stats['total'])}, "
                    f"minimum is {self.min_scenarios_per_required_domain}"
                )
        return failures

    @property
    def coverage_failure_details(self) -> list[dict[str, object]]:
        failures: list[dict[str, object]] = []
        domains = self.domains
        for domain in self.required_domains:
            stats = domains.get(domain)
            if stats is None:
                failures.append(
                    _coverage_failure_detail(
                        domain,
                        "missing",
                        {"passed": 0, "failed": 0, "total": 0},
                        self.min_scenarios_per_required_domain,
                    )
                )
                continue
            if stats["failed"] > 0:
                failures.append(
                    _coverage_failure_detail(
                        domain,
                        "failed",
                        stats,
                        self.min_scenarios_per_required_domain,
                    )
                )
            if stats["total"] < self.min_scenarios_per_required_domain:
                failures.append(
                    _coverage_failure_detail(
                        domain,
                        "below_minimum",
                        stats,
                        self.min_scenarios_per_required_domain,
                    )
                )
        return failures

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "total": self.total,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "domains": self.domains,
            "required_domains": self.required_domains,
            "min_scenarios_per_required_domain": (
                self.min_scenarios_per_required_domain
            ),
            "coverage_failures": self.coverage_failures,
            "coverage_failure_details": self.coverage_failure_details,
            "results": [result.to_dict() for result in self.results],
        }


class ScriptedScenarioLLM:
    def __init__(self, actions: list[ScenarioAction]) -> None:
        self._actions = list(actions)

    def next_action(self, state: RunState, registry: ToolRegistry) -> AgentAction:
        while self._actions:
            action = self._actions.pop(0)
            if isinstance(action, ScriptedConversationAssertion):
                _assert_conversation_contents(state, action.contents)
                continue
            if isinstance(action, ScriptedModelError):
                raise RuntimeError(action.message)
            if isinstance(action, ScriptedInvalidAction):
                return {"type": action.type_name}  # type: ignore[return-value]
            return action
        raise RuntimeError("Scenario actions exhausted")


def _path_exists_or_error(path: Path, label: str) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        raise ValueError(f"{label} could not be checked: {path}") from exc


def _path_is_file_or_error(path: Path, label: str) -> bool:
    try:
        return path.is_file()
    except OSError as exc:
        raise ValueError(f"{label} could not be checked: {path}") from exc


def _utf8_size(value: str) -> int | None:
    encoded = _utf8_bytes(value)
    if encoded is None:
        return None
    return len(encoded)


def _utf8_bytes(value: str) -> bytes | None:
    try:
        return value.encode("utf-8")
    except Exception:
        return None


def _load_yaml_document(path: Path, label: str) -> object:
    if not _path_is_file_or_error(path, label):
        raise ValueError(f"{label} must be a file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"{label} could not be checked: {path}") from exc
    if size > MAX_SCENARIO_FILE_BYTES:
        raise ValueError(f"{label} exceeds {MAX_SCENARIO_FILE_BYTES} bytes: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} could not be read as UTF-8: {path}") from exc
    except OSError as exc:
        raise ValueError(f"{label} could not be read: {path}") from exc
    text_bytes = _utf8_size(text)
    if text_bytes is None:
        raise ValueError(f"{label} could not be inspected: {path}")
    if text_bytes > MAX_SCENARIO_FILE_BYTES:
        raise ValueError(f"{label} exceeds {MAX_SCENARIO_FILE_BYTES} bytes: {path}")
    try:
        loaded = yaml.safe_load(text)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError(f"{label} could not be parsed: {path}") from exc
    return {} if loaded is None else loaded


def load_scenario(path: Path) -> Scenario:
    if not _path_exists_or_error(path, "Scenario file"):
        raise ValueError(f"Scenario file not found: {path}")
    raw = _load_yaml_document(path, "Scenario file")
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario file must contain a mapping: {path}")

    expect = raw.get("expect", {})
    if not isinstance(expect, dict):
        raise ValueError("Scenario expect must be a mapping")
    if not _has_items(expect):
        raise ValueError("Scenario must define expectations")

    goal = raw.get("goal")
    if not isinstance(goal, str):
        raise ValueError("Scenario must define a non-empty goal")
    stripped_goal = _safe_strip(goal)
    if stripped_goal is None:
        raise ValueError("Scenario goal could not be inspected")
    if not stripped_goal and expect.get("status") != "invalid_goal":
        raise ValueError("Scenario must define a non-empty goal")

    raw_actions = raw.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("Scenario must define at least one action")

    files = _parse_files(raw.get("files", {}))
    config_overrides = _parse_config_overrides(raw.get("config", {}))
    user_answers = _parse_user_answers(raw.get("user_answers", []))
    conversation = _parse_conversation(raw.get("conversation", []))
    raw_name = raw.get("name")
    name = raw_name if raw_name is not None else path.stem
    name_text = _safe_text(name)
    if _has_control_characters(name_text):
        raise ValueError("Scenario name cannot contain control characters")
    stripped_name = _safe_strip(name_text)
    if stripped_name == "":
        raise ValueError("Scenario name cannot be blank")
    raw_domain = raw.get("domain")
    domain = raw_domain if raw_domain is not None else "general"
    domain_text = _safe_text(domain)
    if _has_control_characters(domain_text):
        raise ValueError("Scenario domain cannot contain control characters")
    stripped_domain = _safe_strip(domain_text)
    if stripped_domain == "":
        raise ValueError("Scenario domain cannot be blank")
    return Scenario(
        name=stripped_name if stripped_name is not None else name_text,
        domain=stripped_domain if stripped_domain is not None else domain_text,
        goal=goal,
        actions=_parse_actions(raw_actions),
        expect=expect,
        files=files,
        config_overrides=config_overrides,
        user_answers=user_answers,
        conversation=conversation,
    )


def load_scenario_suite(path: Path) -> list[Scenario]:
    scenario_paths = _scenario_paths(path)
    if not scenario_paths:
        raise ValueError(f"No scenario files found: {path}")
    return [load_scenario(scenario_path) for scenario_path in scenario_paths]


def load_scenario_suite_config(path: Path) -> ScenarioSuiteConfig:
    manifest = _suite_manifest_path(path)
    if manifest is None:
        return ScenarioSuiteConfig()

    raw = _load_yaml_document(manifest, "Suite manifest")
    if not isinstance(raw, dict):
        raise ValueError(f"Suite manifest must contain a mapping: {manifest}")
    return ScenarioSuiteConfig(
        required_domains=_parse_required_domains(raw.get("required_domains", [])),
        min_scenarios_per_required_domain=(
            _parse_min_scenarios_per_required_domain(
                raw.get("min_scenarios_per_required_domain")
            )
        ),
    )


def run_scenario(
    scenario: Scenario,
    config: RunConfig,
    registry: ToolRegistry | None = None,
    registry_factory: RegistryFactory | None = None,
) -> ScenarioResult:
    run_id = f"eval-{uuid4()}"
    scenario_actions = _bounded_scenario_actions(scenario.actions)
    scenario_user_answers = _bounded_user_answers(scenario.user_answers)
    scenario_conversation = _bounded_conversation(scenario.conversation)
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
        llm=ScriptedScenarioLLM(scenario_actions),
        registry=scenario_registry,
        logger=logger,
        confirmer=lambda _decision, _action: False,
        run_id=run_id,
        trace=NullTraceSink(),
        ask_user=(
            _scripted_user_answers(scenario_user_answers)
            if _has_items(scenario_user_answers)
            else None
        ),
    )
    state = loop.run(
        scenario.goal,
        conversation=scenario_conversation,
        trace_context={"scenario": scenario.name},
    )
    log_path = Path(logger.path).resolve()
    failures = _compare_expectations(scenario.expect, state, workspace, log_path)
    return ScenarioResult(
        scenario=scenario.name,
        domain=scenario.domain,
        run_id=run_id,
        passed=not failures,
        status=state.final_status,
        answer=state.final_answer,
        reason=state.final_reason,
        observations=len(state.observations),
        failures=failures,
        log_path=log_path,
        workspace=workspace,
    )


def run_scenario_suite(
    scenarios: list[Scenario],
    config: RunConfig,
    registry: ToolRegistry | None = None,
    registry_factory: RegistryFactory | None = None,
    required_domains: list[str] | None = None,
    min_scenarios_per_required_domain: int = 0,
) -> ScenarioSuiteResult:
    results: list[ScenarioResult] = []
    for index, scenario in enumerate(scenarios, 1):
        if index > MAX_SCENARIO_SUITE_FILES:
            raise ValueError(
                "Scenario suite cannot contain more than "
                f"{MAX_SCENARIO_SUITE_FILES} scenarios"
            )
        results.append(
            run_scenario(
                scenario,
                config,
                registry,
                registry_factory=registry_factory,
            )
        )
    return ScenarioSuiteResult(
        results=results,
        required_domains=(
            _parse_required_domains(required_domains)
            if required_domains is not None
            else []
        ),
        min_scenarios_per_required_domain=(
            _parse_min_scenarios_per_required_domain(
                min_scenarios_per_required_domain
            )
        ),
    )


def _scenario_paths(path: Path) -> list[Path]:
    if not _path_exists_or_error(path, "Scenario path"):
        raise ValueError(f"Scenario path not found: {path}")
    if _path_is_file_or_error(path, "Scenario path"):
        if path.name in SUITE_MANIFEST_NAMES:
            return []
        return [path]
    scenario_paths: list[Path] = []
    try:
        for pattern in ("*.yaml", "*.yml"):
            for scenario_path in path.rglob(pattern):
                if scenario_path.name in SUITE_MANIFEST_NAMES:
                    continue
                if len(scenario_paths) >= MAX_SCENARIO_SUITE_FILES:
                    raise ValueError(
                        "Scenario suite cannot contain more than "
                        f"{MAX_SCENARIO_SUITE_FILES} scenarios"
                    )
                scenario_paths.append(scenario_path)
    except OSError as exc:
        raise ValueError(f"Scenario path could not be read: {path}") from exc
    return sorted(scenario_paths, key=lambda item: item.as_posix())


def _suite_manifest_path(path: Path) -> Path | None:
    if _path_is_file_or_error(path, "Suite manifest path"):
        return None
    candidates = [path / name for name in sorted(SUITE_MANIFEST_NAMES)]
    existing = []
    for candidate in candidates:
        try:
            candidate_exists = candidate.exists()
        except OSError as exc:
            raise ValueError(
                f"Suite manifest path could not be checked: {path}"
            ) from exc
        if candidate_exists:
            existing.append(candidate)
    if not existing:
        return None
    if len(existing) > 1:
        raise ValueError(f"Multiple suite manifests found: {path}")
    return existing[0]


def _parse_required_domains(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Suite required_domains must be a list")
    domains: list[str] = []
    for item in raw:
        domain = _inspect_text(item)
        if domain is None:
            raise ValueError("Suite required_domains entries could not be inspected")
        if _has_control_characters(domain):
            raise ValueError(
                "Suite required_domains cannot contain control characters"
            )
        stripped_domain = _safe_strip(domain)
        if stripped_domain is None:
            raise ValueError("Suite required_domains entries could not be inspected")
        if not stripped_domain:
            raise ValueError("Suite required_domains cannot contain empty names")
        domains.append(stripped_domain)
    return domains


def _parse_min_scenarios_per_required_domain(raw: object) -> int:
    if raw is None:
        return 0
    if type(raw) is not int or raw < 0:
        raise ValueError(
            "Suite min_scenarios_per_required_domain must be a non-negative integer"
        )
    return raw


def _scenario_count_text(count: int) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} scenario{suffix}"


def _coverage_failure_detail(
    domain: str,
    reason: str,
    stats: dict[str, int],
    minimum: int,
) -> dict[str, object]:
    return {
        "domain": domain,
        "reason": reason,
        "passed": stats["passed"],
        "failed": stats["failed"],
        "total": stats["total"],
        "minimum": minimum,
    }


def _parse_files(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("Scenario files must be a mapping")
    files: dict[str, str] = {}
    for index, (path, content) in enumerate(raw.items(), 1):
        if index > MAX_SCENARIO_WORKSPACE_FILES:
            raise ValueError(
                "Scenario files cannot contain more than "
                f"{MAX_SCENARIO_WORKSPACE_FILES} entries"
            )
        path_text = _safe_text(path)
        files[path_text] = _parse_file_content(path_text, content)
    return files


def _parse_file_content(path: str, raw: object) -> str:
    if not isinstance(raw, dict):
        return _safe_text(raw)
    repeat = raw.get("repeat")
    count = raw.get("count")
    if not isinstance(repeat, str):
        raise ValueError(f"Scenario file {path} repeat must be a string")
    if type(count) is not int or count < 0:
        raise ValueError(
            f"Scenario file {path} count must be a non-negative integer"
        )
    repeat_bytes = _utf8_size(repeat)
    if repeat_bytes is None:
        raise ValueError(f"Scenario file {path} repeat could not be inspected")
    if repeat_bytes * count > MAX_SCENARIO_FILE_BYTES:
        raise ValueError(
            f"Scenario file {path} exceeds {MAX_SCENARIO_FILE_BYTES} bytes"
        )
    return repeat * count


def _parse_config_overrides(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError("Scenario config must be a mapping")
    return dict(raw)


def _parse_user_answers(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Scenario user_answers must be a list")
    return _bounded_user_answers(_safe_text(answer) for answer in raw)


def _parse_conversation(raw: object) -> list[ConversationMessage]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Scenario conversation must be a list")

    conversation: list[ConversationMessage] = []
    for index, item in enumerate(_bounded_raw_conversation(raw), 1):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario conversation turn {index} must be a mapping")
        role = item.get("role", "user")
        content = item.get("content", "")
        metadata = item.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Scenario conversation turn {index} metadata must be a mapping"
            )
        conversation.append(
            ConversationMessage(
                role=_safe_text(role),
                content=_safe_text(content),
                metadata=dict(metadata),
            )
        )
    return conversation


def _bounded_raw_conversation(raw: Iterable[object]) -> list[object]:
    bounded: list[object] = []
    for index, item in enumerate(raw, 1):
        if index > MAX_SCENARIO_CONVERSATION_TURNS:
            raise ValueError(
                "Scenario conversation cannot contain more than "
                f"{MAX_SCENARIO_CONVERSATION_TURNS} turns"
            )
        bounded.append(item)
    return bounded


def _bounded_conversation(
    conversation: Iterable[ConversationMessage],
) -> list[ConversationMessage]:
    bounded: list[ConversationMessage] = []
    for index, turn in enumerate(conversation, 1):
        if index > MAX_SCENARIO_CONVERSATION_TURNS:
            raise ValueError(
                "Scenario conversation cannot contain more than "
                f"{MAX_SCENARIO_CONVERSATION_TURNS} turns"
            )
        bounded.append(turn)
    return bounded


def _scripted_user_answers(answers: list[str]) -> Callable[[str], str]:
    remaining = iter(answers)

    def ask_user(_question: str) -> str:
        try:
            return next(remaining)
        except StopIteration as exc:
            raise RuntimeError("Scenario user answers exhausted") from exc

    return ask_user


def _bounded_user_answers(answers: Iterable[str]) -> list[str]:
    bounded: list[str] = []
    for index, answer in enumerate(answers, 1):
        if index > MAX_SCENARIO_USER_ANSWERS:
            raise ValueError(
                "Scenario user_answers cannot contain more than "
                f"{MAX_SCENARIO_USER_ANSWERS} entries"
            )
        bounded.append(_safe_text(answer))
    return bounded


def _has_items(items: Iterable[object]) -> bool:
    for _item in items:
        return True
    return False


def _apply_config_overrides(
    config: RunConfig,
    overrides: dict[str, object],
) -> RunConfig:
    if not _has_items(overrides):
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
    if not _has_items(scenario.files):
        return config.workspace.resolve()

    workspace = (config.logs_dir / "workspaces" / run_id).expanduser().resolve()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Scenario workspace could not be prepared: {workspace}"
        ) from exc
    for index, (requested_path, content) in enumerate(scenario.files.items(), 1):
        if index > MAX_SCENARIO_WORKSPACE_FILES:
            raise ValueError(
                "Scenario files cannot contain more than "
                f"{MAX_SCENARIO_WORKSPACE_FILES} entries"
            )
        try:
            target = resolve_workspace_path(workspace, requested_path)
        except PathSecurityError as exc:
            raise ValueError(_safe_text(exc)) from exc
        try:
            parent_exists = target.parent.exists()
        except OSError as exc:
            raise ValueError(
                f"Scenario file parent could not be checked: {requested_path}"
            ) from exc
        if parent_exists:
            try:
                parent_is_directory = target.parent.is_dir()
            except OSError as exc:
                raise ValueError(
                    f"Scenario file parent could not be checked: {requested_path}"
                ) from exc
            if not parent_is_directory:
                raise ValueError(
                    f"Scenario file parent is not a directory: {requested_path}"
                )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"Scenario file parent could not be created: {requested_path}"
            ) from exc
        try:
            target_exists = target.exists()
        except OSError as exc:
            raise ValueError(
                f"Scenario file path could not be checked: {requested_path}"
            ) from exc
        if target_exists:
            try:
                target_is_file = target.is_file()
            except OSError as exc:
                raise ValueError(
                    f"Scenario file path could not be checked: {requested_path}"
                ) from exc
            if not target_is_file:
                raise ValueError(f"Scenario file path is not a file: {requested_path}")
        content_bytes = _utf8_bytes(content)
        if content_bytes is None:
            raise ValueError(
                f"Scenario file {requested_path} content could not be inspected"
            )
        if len(content_bytes) > MAX_SCENARIO_FILE_BYTES:
            raise ValueError(
                f"Scenario file {requested_path} exceeds "
                f"{MAX_SCENARIO_FILE_BYTES} bytes"
            )
        try:
            target.write_bytes(content_bytes)
        except OSError as exc:
            raise ValueError(
                f"Scenario file could not be written: {requested_path}"
            ) from exc
    return workspace


def _parse_action(raw: object, index: int) -> ScenarioAction:
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario action {index + 1} must be a mapping")

    action_type = raw.get("type")
    if action_type == "final":
        text = raw.get("text", "")
        status = raw.get("status", "completed")
        return FinalAction(text=_safe_text(text), status=_safe_text(status))

    if action_type == "tool":
        return _parse_tool_call(raw, f"Scenario action {index + 1}", index + 1)

    if action_type == "empty_batch":
        return ToolCallBatchAction(tool_calls=[])

    if action_type == "invalid_batch_member":
        return ToolCallBatchAction(
            tool_calls=[{"type": "unexpected"}],  # type: ignore[list-item]
        )

    if action_type == "batch":
        raw_calls = raw["calls"] if "calls" in raw else raw.get("tool_calls")
        if not isinstance(raw_calls, list) or not _has_items(raw_calls):
            raise ValueError(
                f"Scenario action {index + 1} batch must define calls",
            )
        bounded_calls = _bounded_batch_calls(raw_calls, index + 1)
        return ToolCallBatchAction(
            tool_calls=[
                _parse_tool_call(
                    call,
                    f"Scenario action {index + 1} batch call {call_index + 1}",
                    index + 1,
                    call_index + 1,
                )
                for call_index, call in enumerate(bounded_calls)
            ],
        )

    if action_type == "error":
        return ScriptedModelError(message=_safe_text(raw.get("message", "")))

    if action_type == "invalid":
        return ScriptedInvalidAction(
            type_name=_safe_text(raw.get("type_name", "unexpected"))
        )

    if action_type == "assert_conversation":
        return ScriptedConversationAssertion(
            contents=_parse_string_list(
                raw.get("contents", []),
                f"Scenario action {index + 1} contents",
            )
        )

    raise ValueError(f"Unknown scenario action type: {action_type}")


def _parse_actions(raw_actions: list[object]) -> list[ScenarioAction]:
    return [
        _parse_action(raw_action, index)
        for index, raw_action in enumerate(
            _bounded_scenario_actions(raw_actions),
        )
    ]


def _bounded_scenario_actions(
    actions: Iterable[ScenarioAction],
) -> list[ScenarioAction]:
    bounded: list[ScenarioAction] = []
    for index, action in enumerate(actions, 1):
        if index > MAX_SCENARIO_ACTIONS:
            raise ValueError(
                "Scenario cannot contain more than "
                f"{MAX_SCENARIO_ACTIONS} actions"
            )
        bounded.append(action)
    return bounded


def _bounded_batch_calls(
    calls: Iterable[object],
    action_number: int,
) -> list[object]:
    bounded: list[object] = []
    for index, call in enumerate(calls, 1):
        if index > MAX_SCENARIO_BATCH_CALLS:
            raise ValueError(
                f"Scenario action {action_number} batch cannot contain more than "
                f"{MAX_SCENARIO_BATCH_CALLS} calls"
            )
        bounded.append(call)
    return bounded


def _parse_string_list(raw: object, label: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list")
    return [_safe_text(item) for item in raw]


def _assert_conversation_contents(
    state: RunState,
    expected_contents: list[str],
) -> None:
    actual_contents = [turn.content for turn in state.conversation]
    if actual_contents != expected_contents:
        raise RuntimeError(
            "Conversation assertion failed: "
            f"expected {_safe_repr(expected_contents)} "
            f"but got {_safe_repr(actual_contents)}"
        )


def _parse_tool_call(
    raw: object,
    context: str,
    action_number: int,
    call_number: int | None = None,
) -> ToolCallAction:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    raw_tool_name = raw.get("tool")
    tool_name = raw_tool_name if raw_tool_name is not None else raw.get("tool_name")
    if not isinstance(tool_name, str):
        raise ValueError(f"{context} must define tool")
    stripped_tool_name = _safe_strip(tool_name)
    if stripped_tool_name is None:
        raise ValueError(f"{context} tool could not be inspected")
    if not stripped_tool_name:
        raise ValueError(f"{context} must define tool")
    arguments = raw.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"{context} arguments must be a mapping")
    raw_call_id = raw.get("call_id")
    call_id = (
        raw_call_id
        if raw_call_id is not None
        else _default_call_id(action_number, call_number)
    )
    return ToolCallAction(
        tool_name=stripped_tool_name,
        arguments=dict(arguments),
        call_id=_safe_text(call_id),
    )


def _default_call_id(action_number: int, call_number: int | None) -> str:
    if call_number is None:
        return f"scenario-call-{action_number}"
    return f"scenario-call-{action_number}-{call_number}"


def _compare_expectations(
    expect: dict[str, object],
    state: RunState,
    workspace: Path,
    log_path: Path,
) -> list[str]:
    failures: list[str] = []
    _expect_equal(failures, "status", expect, state.final_status)
    _expect_equal(failures, "answer", expect, state.final_answer)
    _expect_equal(failures, "reason", expect, state.final_reason)
    if "observations" in expect and expect["observations"] != len(state.observations):
        failures.append(
            "observations expected "
            f"{_safe_repr(expect['observations'])} "
            f"but got {_safe_repr(len(state.observations))}"
        )
    _expect_observations(failures, expect.get("observations_detail"), state)
    _expect_files(failures, expect.get("files"), workspace)
    _expect_events(failures, expect.get("events"), log_path)
    return failures


def _expect_equal(
    failures: list[str],
    key: str,
    expect: dict[str, Any],
    actual: object,
) -> None:
    if key in expect and expect[key] != actual:
        failures.append(
            f"{key} expected {_safe_repr(expect[key])} "
            f"but got {_safe_repr(actual)}"
        )


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
        path_text = _safe_text(requested_path)
        try:
            target = resolve_workspace_path(workspace, path_text)
        except PathSecurityError as exc:
            failures.append(_safe_text(exc))
            continue
        try:
            target_exists = target.exists()
        except OSError:
            failures.append(f"file {path_text} could not be checked")
            continue
        if not target_exists:
            failures.append(f"file {path_text} expected but was missing")
            continue
        try:
            target_is_file = target.is_file()
        except OSError:
            failures.append(f"file {path_text} could not be checked")
            continue
        if not target_is_file:
            failures.append(f"file {path_text} expected but was not a file")
            continue
        try:
            target_size = target.stat().st_size
        except OSError:
            failures.append(f"file {path_text} could not be checked")
            continue
        if target_size > MAX_SCENARIO_FILE_BYTES:
            failures.append(
                f"file {path_text} exceeds {MAX_SCENARIO_FILE_BYTES} bytes"
            )
            continue
        try:
            actual_content = target.read_text(encoding="utf-8")
        except OSError:
            failures.append(f"file {path_text} could not be read")
            continue
        except UnicodeDecodeError:
            failures.append(f"file {path_text} could not be decoded as UTF-8")
            continue
        actual_content_bytes = _utf8_bytes(actual_content)
        if actual_content_bytes is None:
            failures.append(f"file {path_text} could not be inspected")
            continue
        if len(actual_content_bytes) > MAX_SCENARIO_FILE_BYTES:
            failures.append(
                f"file {path_text} exceeds {MAX_SCENARIO_FILE_BYTES} bytes"
            )
            continue
        expected_text = _safe_text(expected_content)
        if actual_content != expected_text:
            failures.append(
                f"file {path_text} expected {_safe_repr(expected_text)} "
                f"but got {_safe_repr(actual_content)}"
            )


def _expect_observations(
    failures: list[str],
    raw_observations: object,
    state: RunState,
) -> None:
    if raw_observations is None:
        return
    if not isinstance(raw_observations, list):
        failures.append("observations_detail expectation must be a list")
        return

    for index, raw_expected in enumerate(raw_observations):
        number = index + 1
        if index >= len(state.observations):
            failures.append(f"observation {number} expected but was missing")
            continue
        if not isinstance(raw_expected, dict):
            failures.append(f"observation {number} expectation must be a mapping")
            continue
        observation = state.observations[index]
        _expect_observation_field(
            failures,
            number,
            "call_id",
            raw_expected,
            observation.call_id,
        )
        _expect_observation_field(
            failures,
            number,
            "tool",
            raw_expected,
            observation.tool_name,
        )
        _expect_observation_field(
            failures,
            number,
            "success",
            raw_expected,
            observation.result.success,
        )
        _expect_observation_field(
            failures,
            number,
            "summary",
            raw_expected,
            observation.result.summary,
        )
        _expect_observation_field(
            failures,
            number,
            "error",
            raw_expected,
            observation.result.error,
        )
        _expect_payload(failures, number, raw_expected.get("payload"), observation)


def _expect_observation_field(
    failures: list[str],
    number: int,
    field_name: str,
    expected: dict[str, object],
    actual: object,
) -> None:
    if field_name in expected and expected[field_name] != actual:
        failures.append(
            f"observation {number} {field_name} expected "
            f"{_safe_repr(expected[field_name])} but got {_safe_repr(actual)}"
        )


def _expect_payload(
    failures: list[str],
    number: int,
    raw_payload: object,
    observation,
) -> None:
    if raw_payload is None:
        return
    if not isinstance(raw_payload, dict):
        failures.append(f"observation {number} payload expectation must be a mapping")
        return
    for key, expected_value in raw_payload.items():
        actual_value = observation.result.payload.get(key)
        if actual_value != expected_value:
            failures.append(
                f"observation {number} payload.{_safe_text(key)} expected "
                f"{_safe_repr(expected_value)} but got {_safe_repr(actual_value)}"
            )


def _expect_events(
    failures: list[str],
    raw_events: object,
    log_path: Path,
) -> None:
    if raw_events is None:
        return
    if not isinstance(raw_events, list):
        failures.append("events expectation must be a list")
        return

    events = _read_log_events(log_path, failures)
    cursor = 0
    for index, raw_expected in enumerate(raw_events):
        number = index + 1
        if not isinstance(raw_expected, dict):
            failures.append(f"event {number} expectation must be a mapping")
            continue

        found = False
        for actual_index in range(cursor, len(events)):
            if _event_matches(raw_expected, events[actual_index]):
                cursor = actual_index + 1
                found = True
                break
        if not found:
            failures.append(
                f"event {number} expected {_safe_repr(raw_expected)} "
                "but was not found"
            )


def _read_log_events(log_path: Path, failures: list[str]) -> list[dict[str, Any]]:
    try:
        log_exists = log_path.exists()
    except OSError:
        failures.append(f"log file could not be checked: {log_path}")
        return []
    if not log_exists:
        failures.append(f"log file expected but was missing: {log_path}")
        return []
    try:
        log_is_file = log_path.is_file()
    except OSError:
        failures.append(f"log file could not be checked: {log_path}")
        return []
    if not log_is_file:
        failures.append(f"log file expected but was not a file: {log_path}")
        return []
    try:
        log_size = log_path.stat().st_size
    except OSError:
        failures.append(f"log file could not be checked: {log_path}")
        return []
    if log_size > MAX_SCENARIO_FILE_BYTES:
        failures.append(f"log file exceeds {MAX_SCENARIO_FILE_BYTES} bytes: {log_path}")
        return []

    events: list[dict[str, Any]] = []
    try:
        raw_log = log_path.read_bytes()
    except OSError:
        failures.append(f"log file could not be read: {log_path}")
        return []
    if len(raw_log) > MAX_SCENARIO_FILE_BYTES:
        failures.append(f"log file exceeds {MAX_SCENARIO_FILE_BYTES} bytes: {log_path}")
        return []
    raw_lines = raw_log.splitlines()
    for line_number, raw_line in enumerate(raw_lines, 1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"log line {line_number} could not be decoded as UTF-8")
            continue
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"log line {line_number} could not be decoded: {exc.msg}")
            continue
        except RecursionError:
            failures.append(
                f"log line {line_number} could not be decoded: "
                "maximum nesting depth"
            )
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            failures.append(f"log line {line_number} must decode to a mapping")
    return events


def _event_matches(expected: dict[str, object], actual: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                return False
            if not _mapping_contains(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _mapping_contains(
    actual: dict[str, Any],
    expected: dict[str, object],
) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                return False
            if not _mapping_contains(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _safe_text(value: object) -> str:
    text = _inspect_text(value)
    if text is None:
        return UNINSPECTABLE_VALUE
    return text


def _inspect_text(value: object) -> str | None:
    try:
        text = str(value)
    except Exception:
        return None
    if isinstance(text, str):
        return str.__str__(text)
    return None


def _safe_strip(value: str) -> str | None:
    try:
        stripped = value.strip()
    except Exception:
        return None
    if not isinstance(stripped, str):
        return None
    return str.__str__(stripped)


def _has_control_characters(value: str) -> bool:
    return any(ord(character) < 32 for character in value)


def _safe_repr(value: object) -> str:
    return _safe_repr_inner(value, set())


def _safe_repr_inner(value: object, seen: set[int]) -> str:
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in seen:
            return "{...}"
        seen.add(value_id)
        try:
            items = ", ".join(
                f"{_safe_repr_inner(key, seen)}: {_safe_repr_inner(item, seen)}"
                for key, item in value.items()
            )
        except Exception:
            return UNINSPECTABLE_VALUE
        finally:
            seen.remove(value_id)
        return f"{{{items}}}"
    if isinstance(value, list):
        value_id = id(value)
        if value_id in seen:
            return "[...]"
        seen.add(value_id)
        try:
            items = ", ".join(_safe_repr_inner(item, seen) for item in value)
        except Exception:
            return UNINSPECTABLE_VALUE
        finally:
            seen.remove(value_id)
        return f"[{items}]"
    if isinstance(value, tuple):
        value_id = id(value)
        if value_id in seen:
            return "(...)"
        seen.add(value_id)
        try:
            items = [_safe_repr_inner(item, seen) for item in value]
        except Exception:
            return UNINSPECTABLE_VALUE
        finally:
            seen.remove(value_id)
        suffix = "," if len(items) == 1 else ""
        return f"({', '.join(items)}{suffix})"
    try:
        return repr(value)
    except Exception:
        return UNINSPECTABLE_VALUE
