import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, NoReturn
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from agentskeleton.config import load_config
from agentskeleton.core.llm import LLMClient, MissingAPIKeyError
from agentskeleton.core.loop import AgentLoop
from agentskeleton.core.runtime import runtime_metadata as build_runtime_metadata
from agentskeleton.core.session import SessionStore
from agentskeleton.core.state import ConversationMessage
from agentskeleton.core.trace import ConsoleTraceSink, NullTraceSink
from agentskeleton.eval import (
    load_scenario,
    load_scenario_suite,
    load_scenario_suite_config,
    run_scenario,
    run_scenario_suite,
)
from agentskeleton.logging.run_logger import RunLogger
from agentskeleton.policy.permissions import PermissionDecision
from agentskeleton.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from agentskeleton.tools.loading import load_tools_from_modules
from agentskeleton.tools.provenance import (
    registered_tool_provenance,
    tool_schema_hash,
)
from agentskeleton.tools.registry import ToolRegistry
from agentskeleton.tools.shell import ShellTool
from agentskeleton.tools.user import AskUserTool


def configure_streams_for_unicode(*streams) -> None:
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


configure_streams_for_unicode(sys.stdout, sys.stderr)

app = typer.Typer(help="Run a minimal local CLI agent.")
console = Console(markup=False)
MAX_RUN_LOG_BYTES = 2_097_152
MAX_RUN_LOG_TEXT_CHARS = 4_096
MAX_RUN_LOG_EVENT_DEPTH = 64


def build_default_registry(
    enabled_tools: list[str] | None = None,
    tool_modules: list[str] | None = None,
) -> ToolRegistry:
    tools = [
        ListDirTool(),
        ReadFileTool(),
        WriteFileTool(),
        ShellTool(),
        AskUserTool(),
    ]
    tools.extend(load_tools_from_modules(tool_modules))
    registry = ToolRegistry(tools)
    if enabled_tools is None:
        return registry

    by_name = {tool.name: tool for tool in registry.all()}
    selected = []
    for raw_name in enabled_tools:
        name = _enabled_tool_name(raw_name)
        try:
            selected.append(by_name[name])
        except KeyError as exc:
            raise ValueError(f"Unknown enabled tool: {name}") from exc
    return ToolRegistry(selected)


def _load_config_or_exit(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
):
    try:
        return load_config(config_path, overrides)
    except ValueError as exc:
        console.print(f"Configuration error: {exc}", soft_wrap=True)
        raise typer.Exit(1) from exc


def _build_registry_or_exit(
    enabled_tools: list[str] | None,
    tool_modules: list[str] | None,
) -> ToolRegistry:
    try:
        return build_default_registry(enabled_tools, tool_modules)
    except ValueError as exc:
        console.print(f"Configuration error: {exc}", soft_wrap=True)
        raise typer.Exit(1) from exc


def _exit_session_error(exc: ValueError) -> None:
    console.print(f"Session error: {exc}", soft_wrap=True)
    raise typer.Exit(1) from exc


def _exit_run_log_error(exc: ValueError) -> NoReturn:
    console.print(f"Run log error: {exc}", soft_wrap=True)
    raise typer.Exit(1) from exc


def _safe_strip_text(value: str) -> str | None:
    try:
        stripped = value.strip()
    except Exception:
        return None
    if not isinstance(stripped, str):
        return None
    return str.__str__(stripped)


def _validate_goal_or_exit(goal: str) -> str:
    stripped_goal = _safe_strip_text(goal)
    if stripped_goal is None:
        console.print("Goal error: Goal could not be inspected")
        raise typer.Exit(1)
    if not stripped_goal:
        console.print("Goal error: Goal cannot be blank")
        raise typer.Exit(1)
    return str.__str__(goal)


def _print_json(payload: object) -> None:
    console.print(
        json.dumps(_json_output_safe(payload), ensure_ascii=False, indent=2),
        soft_wrap=True,
        markup=False,
    )


def _json_output_safe(
    value: object,
    seen: set[int] | None = None,
    depth: int = 0,
) -> object:
    if depth > MAX_RUN_LOG_EVENT_DEPTH:
        return "<max-depth-exceeded>"
    if value is None or isinstance(value, int | float | bool):
        return value
    if isinstance(value, str):
        return str.__str__(value)

    seen = seen or set()
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return {
                _display_text(key): _json_output_safe(item, seen, depth + 1)
                for key, item in value.items()
            }
        except Exception:
            return "<uninspectable>"
        finally:
            seen.remove(marker)
    if isinstance(value, list | tuple):
        marker = id(value)
        if marker in seen:
            return "<recursive>"
        seen.add(marker)
        try:
            return [_json_output_safe(item, seen, depth + 1) for item in value]
        except Exception:
            return "<uninspectable>"
        finally:
            seen.remove(marker)
    return _display_text(value)


def _tool_implementation_label(tool: object) -> str:
    return type(tool).__qualname__


def _tool_provenance(registered_tool) -> dict[str, object]:
    return registered_tool_provenance(registered_tool)


def _create_run_logger_or_exit(logs_dir: Path, run_id: str) -> RunLogger:
    try:
        return RunLogger(logs_dir, run_id)
    except ValueError as exc:
        _exit_run_log_error(exc)


def _create_llm_or_exit(config, trace):
    try:
        return LLMClient(config, trace=trace)
    except MissingAPIKeyError as exc:
        console.print(str(exc))
        raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"Configuration error: {exc}", soft_wrap=True)
        raise typer.Exit(1) from exc


def _summarize_run_log_or_exit(
    log_path: Path,
    run_id: str | None = None,
    include_events: bool = False,
) -> dict[str, object]:
    try:
        return _summarize_run_log(
            log_path,
            run_id=run_id,
            include_events=include_events,
        )
    except ValueError as exc:
        _exit_run_log_error(exc)


def _registry_from_config(config) -> ToolRegistry:
    return build_default_registry(config.enabled_tools, config.tool_modules)


def _merge_required_domains(
    manifest_domains: list[str],
    option_domains: list[str] | None,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for domains in (manifest_domains, () if option_domains is None else option_domains):
        for domain in domains:
            domain = domain.strip()
            if domain in seen:
                continue
            merged.append(domain)
            seen.add(domain)
    return merged


def _assistant_transcript_content(state) -> str | None:
    if state.final_answer is not None:
        return _display_text(state.final_answer)

    if state.final_status is not None:
        content = f"Run stopped with status {_display_text(state.final_status)}."
        if getattr(state, "final_reason", None) is not None:
            content = f"{content} Reason: {_display_text(state.final_reason)}"
        return _bounded_display_text(content)

    return None


def _assistant_transcript_metadata(
    run_id: str,
    state,
    runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "run_id": run_id,
        "status": state.final_status,
    }
    if getattr(state, "final_reason", None) is not None:
        metadata["reason"] = _display_text(state.final_reason)
    if _has_display_value(runtime):
        metadata["runtime"] = runtime
    return metadata


def _summary_assistant_metadata(
    run_id: str,
    summary: dict[str, object],
) -> dict[str, object]:
    metadata = {
        "run_id": run_id,
        "source": "run_log",
        "status": summary["status"],
    }
    if _has_display_value(summary["reason"]):
        metadata["reason"] = summary["reason"]
    if _has_display_value(summary["last_model_error"]):
        metadata["last_model_error"] = summary["last_model_error"]
    if _has_display_value(summary["last_tool_error"]):
        metadata["last_tool_error"] = summary["last_tool_error"]
    if _has_display_value(summary["last_snapshot"]):
        metadata["last_snapshot"] = summary["last_snapshot"]
    runtime = summary.get("runtime")
    if _has_display_value(runtime):
        metadata["runtime"] = runtime
    return metadata


def _summary_conversation(
    run_id: str,
    summary: dict[str, object],
) -> list[ConversationMessage]:
    goal = summary["goal"]
    if not isinstance(goal, str) or not _has_display_value(goal):
        return []
    goal_text = _display_text(goal)
    conversation = [
        ConversationMessage(
            role="user",
            content=goal_text,
            metadata={"run_id": run_id, "source": "run_log"},
        )
    ]
    assistant_content = _summary_transcript_content(summary)
    if assistant_content:
        conversation.append(
            ConversationMessage(
                role="assistant",
                content=assistant_content,
                metadata=_summary_assistant_metadata(run_id, summary),
            )
        )
    return conversation


@app.command()
def tools(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    loaded = _load_config_or_exit(config, {"enabled_tools": tool})
    registry = _build_registry_or_exit(loaded.enabled_tools, loaded.tool_modules)
    if as_json:
        _print_json(
            {
                "tools": [
                    {
                        **_tool_provenance(registered_tool),
                        "args_schema": registered_tool.args_schema,
                    }
                    for registered_tool in registry.all()
                ]
            }
        )
        return

    table = Table(title="Registered Tools")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Risk")
    table.add_column("Schema")
    table.add_column("Implementation")

    for registered_tool in registry.all():
        table.add_row(
            registered_tool.name,
            registered_tool.description,
            registered_tool.risk,
            tool_schema_hash(registered_tool.args_schema)[:12],
            _tool_implementation_label(registered_tool),
        )

    console.print(table)


@app.command()
def doctor(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    loaded = _load_config_or_exit(config, {"enabled_tools": tool})
    registry = _build_registry_or_exit(loaded.enabled_tools, loaded.tool_modules)
    tool_names = [registered_tool.name for registered_tool in registry.all()]
    runtime = build_runtime_metadata(loaded, registry)
    payload = {
        "status": "ok",
        **runtime,
        "workspace": str(loaded.workspace),
        "logs_dir": str(loaded.logs_dir),
        "tool_count": len(tool_names),
        "tools": tool_names,
        "runtime": runtime,
    }

    if as_json:
        _print_json(payload)
        return

    console.print("Status: ok")
    console.print(f"Model: {payload['model']}")
    console.print(f"Reasoning effort: {payload['reasoning_effort']}")
    console.print(f"Text verbosity: {payload['text_verbosity']}")
    console.print(f"Max steps: {payload['max_steps']}")
    console.print(f"Model retry attempts: {payload['model_retry_attempts']}")
    console.print(f"Permission profile: {payload['permission_profile']}")
    console.print(f"Confirm risky actions: {payload['confirm_risky_actions']}")
    enabled_tools = payload["enabled_tools"]
    enabled_tools_text = (
        "default"
        if enabled_tools is None
        else ", ".join(enabled_tools) or "none"
    )
    tool_modules_text = ", ".join(payload["tool_modules"]) or "none"
    console.print(f"Enabled tools: {enabled_tools_text}")
    console.print(f"Tool modules: {tool_modules_text}")
    console.print(f"Workspace: {payload['workspace']}", soft_wrap=True)
    console.print(f"Logs: {payload['logs_dir']}", soft_wrap=True)
    console.print(f"Tools: {payload['tool_count']}")
    console.print(f"Registered tools: {len(payload['registered_tools'])}")
    for registered_tool in payload["registered_tools"]:
        tool_text = _registered_tool_text(registered_tool)
        if tool_text:
            console.print(f"  {tool_text}", soft_wrap=True)


@app.command(name="eval")
def eval_scenario(
    scenario: Path,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    loaded = _load_config_or_exit(config, {"enabled_tools": tool})
    try:
        result = run_scenario(
            load_scenario(scenario),
            loaded,
            registry_factory=_registry_from_config,
        )
    except ValueError as exc:
        console.print(f"Scenario error: {exc}", soft_wrap=True)
        raise typer.Exit(1) from exc

    payload = result.to_dict()
    if as_json:
        _print_json(payload)
    else:
        console.print(f"Scenario: {payload['scenario']}")
        console.print(f"Run id: {payload['run_id']}")
        console.print(f"Passed: {payload['passed']}")
        console.print(f"Status: {payload['status']}")
        if _has_display_value(payload["reason"]):
            reason = _display_text(payload["reason"])
            console.print(f"Reason: {reason}")
        if _has_display_value(payload["failures"]):
            for failure in payload["failures"]:
                bounded_failure = _display_text(failure)
                console.print(f"Failure: {bounded_failure}")
        if payload.get("workspace"):
            console.print(f"Workspace: {payload['workspace']}", soft_wrap=True)
        console.print(f"Log: {payload['log']}", soft_wrap=True)

    if not result.passed:
        raise typer.Exit(1)


@app.command(name="eval-suite")
def eval_suite(
    path: Path,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    require_domain: Annotated[
        list[str] | None,
        typer.Option("--require-domain"),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    loaded = _load_config_or_exit(config, {"enabled_tools": tool})
    try:
        suite_config = load_scenario_suite_config(path)
        result = run_scenario_suite(
            load_scenario_suite(path),
            loaded,
            registry_factory=_registry_from_config,
            required_domains=_merge_required_domains(
                suite_config.required_domains,
                require_domain,
            ),
            min_scenarios_per_required_domain=(
                suite_config.min_scenarios_per_required_domain
            ),
        )
    except ValueError as exc:
        console.print(f"Scenario error: {exc}", soft_wrap=True)
        raise typer.Exit(1) from exc

    payload = result.to_dict()
    if as_json:
        _print_json(payload)
    else:
        console.print(f"Passed: {payload['passed']}")
        console.print(f"Scenarios: {payload['passed_count']}/{payload['total']}")
        for item in payload["results"]:
            marker = "PASS" if item["passed"] else "FAIL"
            run_id_text = (
                f" run={item['run_id']}" if item.get("run_id") else ""
            )
            console.print(
                f"{marker}: {item['scenario']} status={item['status']}"
                f"{run_id_text}"
            )
            if _has_display_value(item["reason"]):
                reason = _display_text(item["reason"])
                console.print(f"Reason: {reason}")
            for failure in item["failures"]:
                bounded_failure = _display_text(failure)
                console.print(f"Failure: {bounded_failure}")
            if not item["passed"]:
                if item.get("workspace"):
                    console.print(f"Workspace: {item['workspace']}", soft_wrap=True)
                if item.get("log"):
                    console.print(f"Log: {item['log']}", soft_wrap=True)
        for failure in payload["coverage_failures"]:
            bounded_failure = _display_text(failure)
            console.print(f"Coverage failure: {bounded_failure}")

    if not result.passed:
        raise typer.Exit(1)


@app.command()
def run(
    goal: str,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option("--reasoning-effort"),
    ] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
    model_retry_attempts: Annotated[
        int | None,
        typer.Option("--model-retry-attempts"),
    ] = None,
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    session: Annotated[str, typer.Option("--session")] = "default",
    no_session: Annotated[bool, typer.Option("--no-session")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
) -> None:
    goal = _validate_goal_or_exit(goal)
    loaded = _load_config_or_exit(
        config,
        {
            "model": model,
            "base_url": base_url,
            "reasoning_effort": reasoning_effort,
            "max_steps": max_steps,
            "model_retry_attempts": model_retry_attempts,
            "enabled_tools": tool,
        },
    )
    run_id = str(uuid4())
    logger = _create_run_logger_or_exit(loaded.logs_dir, run_id)
    registry = _build_registry_or_exit(loaded.enabled_tools, loaded.tool_modules)
    trace = NullTraceSink() if quiet else ConsoleTraceSink(console)
    session_store = SessionStore(loaded.logs_dir)
    conversation = []
    if not no_session:
        try:
            session_store.refresh_summary(
                session,
                loaded.session_context_turns,
                loaded.session_summary_turns,
            )
            session_state = session_store.load(session)
        except ValueError as exc:
            _exit_session_error(exc)
        conversation = session_state.context_messages()

    llm = _create_llm_or_exit(loaded, trace)

    def confirm(decision: PermissionDecision, action) -> bool:
        reason = _display_text(decision.reason)
        arguments = _display_text(action.arguments)
        console.print(f"Tool requires confirmation: {action.tool_name}")
        console.print(f"Reason: {reason}")
        console.print(f"Arguments: {arguments}")
        return typer.confirm("Allow this action?", default=False)

    loop = AgentLoop(
        config=loaded,
        llm=llm,
        registry=registry,
        logger=logger,
        confirmer=confirm,
        ask_user=lambda question: typer.prompt(question),
        run_id=run_id,
        trace=trace,
    )
    if not no_session:
        try:
            session_store.append_transcript(
                session,
                "user",
                goal,
                {"run_id": run_id},
            )
        except ValueError as exc:
            _exit_session_error(exc)
    state = loop.run(
        goal,
        conversation=conversation,
        trace_context={"session": None if no_session else session},
    )
    assistant_content = _assistant_transcript_content(state)
    if not no_session and assistant_content:
        try:
            session_store.append_transcript(
                session,
                "assistant",
                assistant_content,
                _assistant_transcript_metadata(
                    run_id,
                    state,
                    build_runtime_metadata(loaded, registry),
                ),
            )
        except ValueError as exc:
            _exit_session_error(exc)
    console.print(f"Run id: {run_id}")
    console.print(f"Status: {state.final_status}")
    if getattr(state, "final_reason", None) is not None:
        reason = _display_text(state.final_reason)
        console.print(f"Reason: {reason}")
    if state.final_answer is not None:
        console.print(_display_text(state.final_answer))
    console.print(f"Run log: {logger.path}")


@app.command()
def chat(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option("--reasoning-effort"),
    ] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
    model_retry_attempts: Annotated[
        int | None,
        typer.Option("--model-retry-attempts"),
    ] = None,
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    session: Annotated[str, typer.Option("--session")] = "default",
    no_session: Annotated[bool, typer.Option("--no-session")] = False,
    trace: Annotated[bool, typer.Option("--trace")] = False,
) -> None:
    loaded = _load_config_or_exit(
        config,
        {
            "model": model,
            "base_url": base_url,
            "reasoning_effort": reasoning_effort,
            "max_steps": max_steps,
            "model_retry_attempts": model_retry_attempts,
            "enabled_tools": tool,
        },
    )
    registry = _build_registry_or_exit(loaded.enabled_tools, loaded.tool_modules)
    trace_sink = ConsoleTraceSink(console) if trace else NullTraceSink()
    session_store = SessionStore(loaded.logs_dir)

    llm = _create_llm_or_exit(loaded, trace_sink)

    def confirm(decision: PermissionDecision, action) -> bool:
        reason = _display_text(decision.reason)
        arguments = _display_text(action.arguments)
        console.print(f"Tool requires confirmation: {action.tool_name}")
        console.print(f"Reason: {reason}")
        console.print(f"Arguments: {arguments}")
        return typer.confirm("Allow this action?", default=False)

    console.print("Type /exit or /quit to leave.")
    while True:
        try:
            goal = typer.prompt("agent")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        stripped_goal = _safe_strip_text(goal)
        if stripped_goal is None:
            console.print("Goal error: Goal could not be inspected")
            continue
        if stripped_goal.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if not stripped_goal:
            continue
        goal = str.__str__(goal)

        run_id = str(uuid4())
        logger = _create_run_logger_or_exit(loaded.logs_dir, run_id)
        conversation = []
        if not no_session:
            try:
                session_store.refresh_summary(
                    session,
                    loaded.session_context_turns,
                    loaded.session_summary_turns,
                )
                conversation = session_store.load(session).context_messages()
                session_store.append_transcript(
                    session,
                    "user",
                    goal,
                    {"run_id": run_id},
                )
            except ValueError as exc:
                _exit_session_error(exc)

        loop = AgentLoop(
            config=loaded,
            llm=llm,
            registry=registry,
            logger=logger,
            confirmer=confirm,
            ask_user=lambda question: typer.prompt(question),
            run_id=run_id,
            trace=trace_sink,
        )
        state = loop.run(
            goal,
            conversation=conversation,
            trace_context={"session": None if no_session else session},
        )
        assistant_content = _assistant_transcript_content(state)
        if not no_session and assistant_content:
            try:
                session_store.append_transcript(
                    session,
                    "assistant",
                    assistant_content,
                    _assistant_transcript_metadata(
                        run_id,
                        state,
                        build_runtime_metadata(loaded, registry),
                    ),
                )
            except ValueError as exc:
                _exit_session_error(exc)

        console.print(f"Run id: {run_id}")
        console.print(f"Status: {state.final_status}")
        if getattr(state, "final_reason", None) is not None:
            reason = _display_text(state.final_reason)
            console.print(f"Reason: {reason}")
        if state.final_answer is not None:
            console.print(f"assistant> {_display_text(state.final_answer)}")
        console.print(f"Run log: {logger.path}")


@app.command()
def resume(
    session: Annotated[str, typer.Argument()] = "default",
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option("--reasoning-effort"),
    ] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
    model_retry_attempts: Annotated[
        int | None,
        typer.Option("--model-retry-attempts"),
    ] = None,
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    trace: Annotated[bool, typer.Option("--trace")] = False,
) -> None:
    chat(
        config=config,
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        max_steps=max_steps,
        model_retry_attempts=model_retry_attempts,
        tool=tool,
        session=session,
        no_session=False,
        trace=trace,
    )


@app.command(name="resume-run")
def resume_run(
    run_id: str,
    goal: str,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option("--reasoning-effort"),
    ] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps")] = None,
    model_retry_attempts: Annotated[
        int | None,
        typer.Option("--model-retry-attempts"),
    ] = None,
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
) -> None:
    goal = _validate_goal_or_exit(goal)
    loaded = _load_config_or_exit(
        config,
        {
            "model": model,
            "base_url": base_url,
            "reasoning_effort": reasoning_effort,
            "max_steps": max_steps,
            "model_retry_attempts": model_retry_attempts,
            "enabled_tools": tool,
        },
    )
    try:
        log_path = _find_run_log(loaded.logs_dir, run_id)
    except ValueError as exc:
        _exit_run_log_error(exc)
    if log_path is None:
        console.print(f"Run log not found: {run_id}")
        raise typer.Exit(1)

    summary = _summarize_run_log_or_exit(log_path, run_id=run_id)
    conversation = _summary_conversation(run_id, summary)
    if not conversation:
        console.print(f"Run log has no restorable goal: {run_id}")
        raise typer.Exit(1)

    resumed_run_id = str(uuid4())
    logger = _create_run_logger_or_exit(loaded.logs_dir, resumed_run_id)
    registry = _build_registry_or_exit(loaded.enabled_tools, loaded.tool_modules)
    trace = NullTraceSink() if quiet else ConsoleTraceSink(console)
    llm = _create_llm_or_exit(loaded, trace)

    def confirm(decision: PermissionDecision, action) -> bool:
        reason = _display_text(decision.reason)
        arguments = _display_text(action.arguments)
        console.print(f"Tool requires confirmation: {action.tool_name}")
        console.print(f"Reason: {reason}")
        console.print(f"Arguments: {arguments}")
        return typer.confirm("Allow this action?", default=False)

    loop = AgentLoop(
        config=loaded,
        llm=llm,
        registry=registry,
        logger=logger,
        confirmer=confirm,
        ask_user=lambda question: typer.prompt(question),
        run_id=resumed_run_id,
        trace=trace,
    )
    state = loop.run(
        goal,
        conversation=conversation,
        trace_context={"resumed_run_id": run_id},
    )
    console.print(f"Run id: {resumed_run_id}")
    console.print(f"Resumed from: {run_id}")
    console.print(f"Status: {state.final_status}")
    if getattr(state, "final_reason", None) is not None:
        reason = _display_text(state.final_reason)
        console.print(f"Reason: {reason}")
    if state.final_answer is not None:
        console.print(_display_text(state.final_answer))
    console.print(f"Run log: {logger.path}")


@app.command()
def sessions(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    loaded = _load_config_or_exit(config)
    try:
        summaries = SessionStore(loaded.logs_dir).list_sessions()
    except ValueError as exc:
        _exit_session_error(exc)
    payload = {"sessions": [summary.to_dict() for summary in summaries]}

    if as_json:
        _print_json(payload)
        return

    if not summaries:
        console.print("No sessions found.")
        return

    table = Table(title="Sessions")
    table.add_column("Name")
    table.add_column("Turns")
    table.add_column("Summary")
    for summary in summaries:
        table.add_row(
            _display_text(summary.name),
            _display_text(summary.transcript_turns),
            _display_optional_text(summary.summary),
        )
    console.print(table)


@app.command()
def show_run(
    run_id: str,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    include_events: Annotated[bool, typer.Option("--events")] = False,
) -> None:
    loaded = _load_config_or_exit(config)
    try:
        log_path = _find_run_log(loaded.logs_dir, run_id)
    except ValueError as exc:
        _exit_run_log_error(exc)
    if log_path is None:
        console.print(f"Run log not found: {run_id}")
        raise typer.Exit(1)

    summary = _summarize_run_log_or_exit(
        log_path,
        run_id=run_id,
        include_events=include_events,
    )

    if as_json:
        _print_json(summary)
        return

    console.print(f"Run id: {run_id}")
    console.print(f"Status: {_display_text(summary['status'])}")
    if _has_display_value(summary["reason"]):
        console.print(f"Reason: {_display_text(summary['reason'])}")
    if _has_display_value(summary["goal"]):
        console.print(f"Goal: {_display_text(summary['goal'])}", soft_wrap=True)
    runtime = summary.get("runtime")
    if isinstance(runtime, dict):
        _print_run_runtime(runtime)
    if _has_display_value(summary["session"]):
        console.print(f"Session: {_display_text(summary['session'])}")
    if _has_display_value(summary["started_at"]):
        console.print(f"Started: {_display_text(summary['started_at'])}")
    if _has_display_value(summary["finished_at"]):
        console.print(f"Finished: {_display_text(summary['finished_at'])}")
    if summary["duration_seconds"] is not None:
        console.print(f"Duration: {_display_text(summary['duration_seconds'])}s")
    if _has_display_value(summary["resumed_run_id"]):
        console.print(f"Resumed from: {_display_text(summary['resumed_run_id'])}")
    if summary["conversation_turns"] is not None:
        console.print(
            f"Conversation turns: {_display_text(summary['conversation_turns'])}"
        )
    if summary["steps"] is not None:
        console.print(f"Steps: {_display_text(summary['steps'])}")
    snapshot_summary = _snapshot_summary_text(summary.get("last_snapshot"))
    if snapshot_summary:
        console.print(f"Snapshot: {snapshot_summary}")
    if _has_display_value(summary["model_retries"]):
        console.print(f"Model retries: {_display_text(summary['model_retries'])}")
        retry_detail = _error_detail_text(summary.get("last_model_retry"))
        if retry_detail:
            console.print(f"Last model retry: {retry_detail}")
    model_error_detail = _error_detail_text(summary.get("last_model_error"))
    if model_error_detail:
        console.print(f"Last model error: {model_error_detail}")
    if _has_display_value(summary["tool_calls"]):
        console.print(
            f"Tool calls: {_display_text(summary['tool_calls'])} "
            f"({_display_text(summary['tool_failures'])} failed)"
        )
        if _has_display_value(summary["last_tool_error"]):
            tool_error_detail = _tool_error_detail_text(summary["last_tool_error"])
            if tool_error_detail:
                console.print(
                    f"Last tool error: {tool_error_detail}",
                    soft_wrap=True,
                )
    console.print(f"Log: {_display_text(summary['log'])}", soft_wrap=True)


def _print_run_runtime(runtime: dict[object, object]) -> None:
    labels = (
        ("model", "Model"),
        ("reasoning_effort", "Reasoning effort"),
        ("text_verbosity", "Text verbosity"),
        ("max_steps", "Max steps"),
        ("model_retry_attempts", "Model retry attempts"),
        ("permission_profile", "Permission profile"),
        ("confirm_risky_actions", "Confirm risky actions"),
    )
    for key, label in labels:
        if key in runtime:
            console.print(f"{label}: {_display_text(runtime[key])}")
    if "enabled_tools" in runtime:
        console.print(
            f"Enabled tools: {_runtime_list_text(runtime['enabled_tools'], 'default')}"
        )
    if "tool_modules" in runtime:
        console.print(
            f"Tool modules: {_runtime_list_text(runtime['tool_modules'], 'none')}"
        )
    registered_tools = runtime.get("registered_tools")
    if isinstance(registered_tools, list):
        console.print(f"Registered tools: {len(registered_tools)}")
        for tool in registered_tools:
            tool_text = _registered_tool_text(tool)
            if tool_text:
                console.print(f"  {tool_text}", soft_wrap=True)


def _runtime_list_text(value: object, none_label: str) -> str:
    if value is None:
        return none_label
    if isinstance(value, list):
        return ", ".join(_display_text(item) for item in value) or "none"
    return _display_text(value)


def _registered_tool_text(value: object) -> str | None:
    if not isinstance(value, dict):
        return _display_text(value)
    name = value.get("name")
    if name is None:
        return None
    risk = value.get("risk")
    if risk is None:
        return _display_text(name)
    details = [_display_text(risk)]
    args_schema_hash = value.get("args_schema_hash")
    if args_schema_hash is not None:
        details.append(f"schema {_display_text(args_schema_hash)[:12]}")
    implementation = value.get("implementation")
    if implementation is not None:
        details.append(f"impl {_display_text(implementation)}")
    return f"{_display_text(name)} ({', '.join(details)})"


@app.command(name="list-runs")
def list_runs(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1)] = 20,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    loaded = _load_config_or_exit(config)
    try:
        run_log_paths = _run_log_paths(loaded.logs_dir, limit=limit)
    except ValueError as exc:
        _exit_run_log_error(exc)
    summaries = [
        _summarize_run_log_or_exit(log_path)
        for log_path in run_log_paths
    ]

    if as_json:
        _print_json({"runs": summaries})
        return

    if not summaries:
        console.print("No run logs found.")
        return

    table = Table(title="Recent Runs")
    table.add_column("Run ID")
    table.add_column("Status")
    table.add_column("Steps")
    table.add_column("Retries")
    table.add_column("Tool Fail")
    table.add_column("Goal")
    table.add_column("Log")
    for summary in summaries:
        table.add_row(
            _display_text(summary["run_id"]),
            _display_text(summary["status"]),
            "" if summary["steps"] is None else _display_text(summary["steps"]),
            _display_text(summary["model_retries"]),
            (
                f"{_display_text(summary['tool_failures'])}/"
                f"{_display_text(summary['tool_calls'])}"
            ),
            _display_optional_text(summary["goal"]),
            _display_text(summary["log"]),
        )
    console.print(table)
    for summary in summaries:
        started_at = summary.get("started_at")
        finished_at = summary.get("finished_at")
        duration_seconds = summary.get("duration_seconds")
        run_id = _display_text(summary["run_id"])
        if _has_display_value(started_at) and _has_display_value(finished_at):
            duration_text = (
                f" duration: {_display_text(duration_seconds)}s"
                if duration_seconds is not None
                else ""
            )
            console.print(
                f"Run {run_id} started: {_display_text(started_at)} "
                f"finished: {_display_text(finished_at)}{duration_text}",
                soft_wrap=True,
            )
        elif _has_display_value(started_at):
            console.print(
                f"Run {run_id} started: {_display_text(started_at)}",
                soft_wrap=True,
            )
        resumed_run_id = summary.get("resumed_run_id")
        if _has_display_value(resumed_run_id):
            console.print(
                f"Run {run_id} resumed from: {_display_text(resumed_run_id)}"
            )


@app.command(name="restore-run")
def restore_run(
    run_id: str,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    session: Annotated[str, typer.Option("--session")] = "default",
) -> None:
    loaded = _load_config_or_exit(config)
    try:
        log_path = _find_run_log(loaded.logs_dir, run_id)
    except ValueError as exc:
        _exit_run_log_error(exc)
    if log_path is None:
        console.print(f"Run log not found: {run_id}")
        raise typer.Exit(1)

    summary = _summarize_run_log_or_exit(log_path, run_id=run_id)
    goal = summary["goal"]
    if not isinstance(goal, str) or not _has_display_value(goal):
        console.print(f"Run log has no restorable goal: {run_id}")
        raise typer.Exit(1)
    goal_text = _display_text(goal)

    store = SessionStore(loaded.logs_dir)
    try:
        if _session_has_restored_run(store, session, run_id):
            console.print(f"Run {run_id} is already restored in session {session}")
            return
    except ValueError as exc:
        _exit_session_error(exc)

    try:
        store.append_transcript(
            session,
            "user",
            goal_text,
            {"run_id": run_id, "source": "run_log"},
        )
        assistant_content = _summary_transcript_content(summary)
        if assistant_content:
            store.append_transcript(
                session,
                "assistant",
                assistant_content,
                _summary_assistant_metadata(run_id, summary),
            )
    except ValueError as exc:
        _exit_session_error(exc)

    console.print(f"Restored run {run_id} into session {session}")


def _session_has_restored_run(
    store: SessionStore,
    session: str,
    run_id: str,
) -> bool:
    return any(
        turn.metadata.get("source") == "run_log"
        and turn.metadata.get("run_id") == run_id
        for turn in store.load(session).transcript
    )


def main() -> None:
    app()


def _find_run_log(logs_dir: Path, run_id: str) -> Path | None:
    filename = f"{run_id}.jsonl"
    try:
        matches = sorted(
            (path for path in logs_dir.glob("*/*.jsonl") if path.name == filename),
            reverse=True,
        )
    except OSError as exc:
        raise ValueError(f"Run log directory could not be read: {logs_dir}") from exc
    return matches[0].resolve() if matches else None


def _run_log_paths(logs_dir: Path, limit: int | None = None) -> list[Path]:
    try:
        paths = (path for path in logs_dir.glob("*/*.jsonl") if path.is_file())
        sorted_paths = sorted(
            paths,
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
            reverse=True,
        )
        if limit is None:
            return sorted_paths
        return sorted_paths[:limit]
    except OSError as exc:
        raise ValueError(f"Run log directory could not be read: {logs_dir}") from exc


def _summarize_run_log(
    log_path: Path,
    run_id: str | None = None,
    include_events: bool = False,
) -> dict[str, object]:
    events = _read_run_events(log_path)
    start_event = next(
        (event for event in events if event.get("type") == "run_started"),
        None,
    )
    final_event = next(
        (event for event in reversed(events) if event.get("type") == "run_finished"),
        None,
    )
    start_payload = _event_payload(start_event)
    final_payload = _event_payload(final_event)
    step = final_event.get("step") if final_event else None
    tool_events = [
        event
        for event in events
        if event.get("type") == "tool_finished"
        and isinstance(event.get("payload"), dict)
    ]
    model_retry_events = [
        event
        for event in events
        if event.get("type") == "model_retry"
        and isinstance(event.get("payload"), dict)
    ]
    model_error_events = [
        event
        for event in events
        if event.get("type") == "run_error"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("status") == "model_error"
    ]
    snapshot_events = [
        event
        for event in events
        if event.get("type") == "run_snapshot"
        and isinstance(event.get("payload"), dict)
    ]
    failed_tool_payloads = [
        event["payload"]
        for event in tool_events
        if event["payload"].get("success") is False
    ]
    last_model_retry = None
    if model_retry_events:
        last_retry_payload = model_retry_events[-1]["payload"]
        last_model_retry = {
            "attempt": last_retry_payload.get("attempt"),
            "next_attempt": last_retry_payload.get("next_attempt"),
            "max_attempts": last_retry_payload.get("max_attempts"),
            "error_type": _run_log_text_value(last_retry_payload.get("error_type")),
            "error": _run_log_text_value(last_retry_payload.get("error")),
        }
    last_model_error = None
    if model_error_events:
        last_error_payload = model_error_events[-1]["payload"]
        last_model_error = {
            "attempt": last_error_payload.get("attempt"),
            "max_attempts": last_error_payload.get("max_attempts"),
            "error_type": _run_log_text_value(last_error_payload.get("error_type")),
            "error": _run_log_text_value(last_error_payload.get("error")),
        }
    last_tool_error = None
    if failed_tool_payloads:
        last_failed = failed_tool_payloads[-1]
        last_tool_error = {
            "tool_name": _run_log_text_value(last_failed.get("tool_name")),
            "summary": _run_log_text_value(last_failed.get("summary")),
            "error": _run_log_text_value(last_failed.get("error")),
        }
    last_snapshot = None
    if snapshot_events:
        bounded_snapshot = _bounded_run_log_event(snapshot_events[-1])
        snapshot_payload = bounded_snapshot.get("payload")
        if isinstance(snapshot_payload, dict):
            last_snapshot = snapshot_payload
    resolved_run_id = run_id
    if resolved_run_id is None and final_event and final_event.get("run_id"):
        resolved_run_id = str(final_event["run_id"])
    if resolved_run_id is None and start_event and start_event.get("run_id"):
        resolved_run_id = str(start_event["run_id"])
    if resolved_run_id is None:
        resolved_run_id = log_path.stem
    started_at = _run_log_text_value(
        start_event.get("timestamp") if start_event else None
    )
    finished_at = _run_log_text_value(
        final_event.get("timestamp") if final_event else None
    )
    summary = {
        "run_id": resolved_run_id,
        "goal": _run_log_text_value(start_payload.get("goal")),
        "workspace": start_payload.get("workspace"),
        "session": start_payload.get("session"),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": _run_duration_seconds(started_at, finished_at),
        "resumed": start_payload.get("resumed"),
        "resumed_run_id": _run_log_text_value(start_payload.get("resumed_run_id")),
        "conversation_turns": start_payload.get("conversation_turns"),
        "status": final_payload.get("status", "unknown"),
        "reason": _run_log_text_value(final_payload.get("reason")),
        "answer": _run_log_text_value(final_payload.get("answer")),
        "steps": step,
        "model_retries": len(model_retry_events),
        "last_model_retry": last_model_retry,
        "last_model_error": last_model_error,
        "tool_calls": len(tool_events),
        "tool_failures": len(failed_tool_payloads),
        "last_tool_error": last_tool_error,
        "last_snapshot": last_snapshot,
        "log": str(log_path.resolve()),
    }
    runtime = _run_runtime_summary(start_payload)
    if runtime:
        summary["runtime"] = runtime
    if include_events:
        summary["events"] = [_bounded_run_log_event(event) for event in events]
    return summary


def _run_runtime_summary(start_payload: dict[str, Any]) -> dict[str, object]:
    runtime_keys = (
        "model",
        "reasoning_effort",
        "text_verbosity",
        "max_steps",
        "model_retry_attempts",
        "permission_profile",
        "confirm_risky_actions",
        "enabled_tools",
        "tool_modules",
        "registered_tools",
    )
    return {
        key: _bounded_run_log_event(_run_log_text_value(start_payload[key]))
        for key in runtime_keys
        if key in start_payload
    }


def _event_payload(event: dict[str, Any] | None) -> dict[str, Any]:
    if event is None:
        return {}
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _run_duration_seconds(started_at: object, finished_at: object) -> float | None:
    if not isinstance(started_at, str) or not isinstance(finished_at, str):
        return None
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
        duration = (finished - started).total_seconds()
    except (TypeError, ValueError):
        return None
    return duration if duration >= 0 else None


def _run_log_text_value(value: object) -> object:
    if isinstance(value, dict) and value.get("truncated") is True:
        preview = value.get("preview")
        if not isinstance(preview, str):
            return value
        byte_count = value.get("bytes")
        if isinstance(byte_count, int):
            return f"{preview} (truncated, {byte_count} bytes)"
        return f"{preview} (truncated)"
    if isinstance(value, str):
        return _bounded_run_log_text(value)
    return value


def _bounded_run_log_text(value: str) -> str:
    value = str.__str__(value)
    if len(value) <= MAX_RUN_LOG_TEXT_CHARS:
        return value
    omitted = len(value) - MAX_RUN_LOG_TEXT_CHARS
    return f"{value[:MAX_RUN_LOG_TEXT_CHARS]}\n[truncated {omitted} characters]"


def _bounded_display_text(value: str) -> str:
    return _bounded_run_log_text(value)


def _display_text(value: object) -> str:
    try:
        text = str(value)
    except Exception:
        text = "<uninspectable>"
    text = str.__str__(text)
    return _bounded_display_text(text)


def _has_display_value(value: object) -> bool:
    try:
        return bool(value)
    except Exception:
        return True


def _display_optional_text(value: object) -> str:
    if value is None:
        return ""
    return _display_text(value)


def _enabled_tool_name(value: object) -> str:
    if isinstance(value, str):
        return str.__str__(value)
    return _display_text(value)


def _bounded_run_log_event(value: Any, depth: int = 0) -> Any:
    if depth > MAX_RUN_LOG_EVENT_DEPTH:
        return "<max-depth-exceeded>"
    if isinstance(value, str):
        return _bounded_run_log_text(value)
    if isinstance(value, list):
        return [_bounded_run_log_event(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            _display_text(key): _bounded_run_log_event(item, depth + 1)
            for key, item in value.items()
        }
    return value


def _summary_transcript_content(summary: dict[str, object]) -> str | None:
    answer = summary.get("answer")
    if isinstance(answer, str) and _has_display_value(answer):
        answer_text = _display_text(answer)
        snapshot_detail = _snapshot_detail_text(summary.get("last_snapshot"))
        if snapshot_detail:
            return _bounded_display_text(
                f"{answer_text} Last run snapshot: {snapshot_detail}"
            )
        return _bounded_display_text(answer_text)

    status = summary.get("status")
    if isinstance(status, str) and _has_display_value(status):
        content = f"Run stopped with status {_display_text(status)}."
        reason = summary.get("reason")
        if isinstance(reason, str) and _has_display_value(reason):
            content = f"{content} Reason: {_display_text(reason)}"
        model_error_detail = _error_detail_text(summary.get("last_model_error"))
        if model_error_detail:
            content = f"{content} Last model error: {model_error_detail}"
        tool_error_detail = _tool_error_detail_text(summary.get("last_tool_error"))
        if tool_error_detail:
            content = f"{content} Last tool error: {tool_error_detail}"
        snapshot_detail = _snapshot_detail_text(summary.get("last_snapshot"))
        if snapshot_detail:
            content = f"{content} Last run snapshot: {snapshot_detail}"
        return _bounded_display_text(content)

    return None


def _snapshot_detail_text(snapshot: object) -> str | None:
    if not isinstance(snapshot, dict):
        return None
    try:
        text = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        text = _display_text(snapshot)
    return _bounded_display_text(text)


def _snapshot_summary_text(snapshot: object) -> str | None:
    if not isinstance(snapshot, dict):
        return None
    step_count = snapshot.get("step_count")
    observations = snapshot.get("observations")
    if not isinstance(observations, list):
        return None
    return f"step {_display_text(step_count)}, observations {len(observations)}"


def _error_detail_text(error_detail: object) -> str | None:
    if not isinstance(error_detail, dict):
        return None
    error_type_value = error_detail.get("error_type")
    error_value = error_detail.get("error")
    attempt_value = error_detail.get("attempt")
    max_attempts_value = error_detail.get("max_attempts")
    attempt_detail = None
    if attempt_value is not None and max_attempts_value is not None:
        attempt = _display_text(attempt_value)
        max_attempts = _display_text(max_attempts_value)
        attempt_detail = f" (attempt {attempt}/{max_attempts})"
    if error_type_value is not None and error_value is not None:
        error_type = _display_text(error_type_value)
        error = _display_text(error_value)
        return f"{error_type} - {error}{attempt_detail or ''}"
    if error_type_value is not None:
        return f"{_display_text(error_type_value)}{attempt_detail or ''}"
    if error_value is not None:
        return f"{_display_text(error_value)}{attempt_detail or ''}"
    return None


def _tool_error_detail_text(error_detail: object) -> str | None:
    if not isinstance(error_detail, dict):
        return None
    tool_name_value = error_detail.get("tool_name")
    summary_value = error_detail.get("summary")
    error_value = error_detail.get("error")
    detail_value = summary_value if summary_value is not None else error_value
    if tool_name_value is not None and detail_value is not None:
        tool_name = _display_text(tool_name_value)
        detail = _display_text(detail_value)
        return f"{tool_name} - {detail}"
    if tool_name_value is not None:
        return _display_text(tool_name_value)
    if detail_value is not None:
        return _display_text(detail_value)
    return None


def _read_run_events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        raise ValueError(f"Run log path is not a file: {log_path}")

    events: list[dict[str, Any]] = []
    try:
        if log_path.stat().st_size > MAX_RUN_LOG_BYTES:
            raise ValueError(f"Run log exceeds {MAX_RUN_LOG_BYTES} bytes: {log_path}")
        raw_log = log_path.read_bytes()
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"Run log could not be read: {log_path}") from exc
    if len(raw_log) > MAX_RUN_LOG_BYTES:
        raise ValueError(f"Run log exceeds {MAX_RUN_LOG_BYTES} bytes: {log_path}")
    raw_lines = raw_log.splitlines()
    for raw_line in raw_lines:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if line.strip():
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, RecursionError):
                continue
            if isinstance(event, dict):
                events.append(event)
    return events
