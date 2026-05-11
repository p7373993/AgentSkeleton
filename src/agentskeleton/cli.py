import json
import sys
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from agentskeleton.config import load_config
from agentskeleton.core.llm import LLMClient, MissingAPIKeyError
from agentskeleton.core.loop import AgentLoop
from agentskeleton.core.session import SessionStore
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
console = Console()


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
    for name in enabled_tools:
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


def _create_run_logger_or_exit(logs_dir: Path, run_id: str) -> RunLogger:
    try:
        return RunLogger(logs_dir, run_id)
    except ValueError as exc:
        console.print(f"Run log error: {exc}", soft_wrap=True)
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
        console.print(f"Run log error: {exc}", soft_wrap=True)
        raise typer.Exit(1) from exc


def _registry_from_config(config) -> ToolRegistry:
    return build_default_registry(config.enabled_tools, config.tool_modules)


def _merge_required_domains(
    manifest_domains: list[str],
    option_domains: list[str] | None,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for domain in [*manifest_domains, *(option_domains or [])]:
        if domain in seen:
            continue
        merged.append(domain)
        seen.add(domain)
    return merged


def _assistant_transcript_content(state) -> str | None:
    if state.final_answer:
        return state.final_answer

    if state.final_status:
        content = f"Run stopped with status {state.final_status}."
        if getattr(state, "final_reason", None):
            content = f"{content} Reason: {state.final_reason}"
        return content

    return None


def _assistant_transcript_metadata(run_id: str, state) -> dict[str, object]:
    metadata: dict[str, object] = {
        "run_id": run_id,
        "status": state.final_status,
    }
    if getattr(state, "final_reason", None):
        metadata["reason"] = state.final_reason
    return metadata


@app.command()
def tools(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    loaded = _load_config_or_exit(config, {"enabled_tools": tool})
    registry = _build_registry_or_exit(loaded.enabled_tools, loaded.tool_modules)
    if as_json:
        console.print(
            json.dumps(
                {
                    "tools": [
                        {
                            "name": registered_tool.name,
                            "description": registered_tool.description,
                            "risk": registered_tool.risk,
                        }
                        for registered_tool in registry.all()
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            soft_wrap=True,
        )
        return

    table = Table(title="Registered Tools")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Risk")

    for registered_tool in registry.all():
        table.add_row(
            registered_tool.name,
            registered_tool.description,
            registered_tool.risk,
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
    payload = {
        "status": "ok",
        "workspace": str(loaded.workspace),
        "logs_dir": str(loaded.logs_dir),
        "tool_count": len(tool_names),
        "tools": tool_names,
    }

    if as_json:
        console.print(json.dumps(payload, ensure_ascii=False, indent=2), soft_wrap=True)
        return

    console.print("Status: ok")
    console.print(f"Workspace: {payload['workspace']}", soft_wrap=True)
    console.print(f"Logs: {payload['logs_dir']}", soft_wrap=True)
    console.print(f"Tools: {payload['tool_count']}")


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
        console.print(json.dumps(payload, ensure_ascii=False, indent=2), soft_wrap=True)
    else:
        console.print(f"Scenario: {payload['scenario']}")
        console.print(f"Passed: {payload['passed']}")
        console.print(f"Status: {payload['status']}")
        if payload["reason"]:
            console.print(f"Reason: {payload['reason']}")
        if payload["failures"]:
            for failure in payload["failures"]:
                console.print(f"Failure: {failure}")
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
        )
    except ValueError as exc:
        console.print(f"Scenario error: {exc}", soft_wrap=True)
        raise typer.Exit(1) from exc

    payload = result.to_dict()
    if as_json:
        console.print(json.dumps(payload, ensure_ascii=False, indent=2), soft_wrap=True)
    else:
        console.print(f"Passed: {payload['passed']}")
        console.print(f"Scenarios: {payload['passed_count']}/{payload['total']}")
        for item in payload["results"]:
            marker = "PASS" if item["passed"] else "FAIL"
            console.print(f"{marker}: {item['scenario']} status={item['status']}")
            if item["reason"]:
                console.print(f"Reason: {item['reason']}")
            for failure in item["failures"]:
                console.print(f"Failure: {failure}")
        for failure in payload["coverage_failures"]:
            console.print(f"Coverage failure: {failure}")

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
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    session: Annotated[str, typer.Option("--session")] = "default",
    no_session: Annotated[bool, typer.Option("--no-session")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
) -> None:
    loaded = _load_config_or_exit(
        config,
        {
            "model": model,
            "base_url": base_url,
            "reasoning_effort": reasoning_effort,
            "max_steps": max_steps,
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

    try:
        llm = LLMClient(loaded, trace=trace)
    except MissingAPIKeyError as exc:
        console.print(str(exc))
        raise typer.Exit(1) from exc

    def confirm(decision: PermissionDecision, action) -> bool:
        console.print(f"Tool requires confirmation: {action.tool_name}")
        console.print(f"Reason: {decision.reason}")
        console.print(f"Arguments: {action.arguments}")
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
                _assistant_transcript_metadata(run_id, state),
            )
        except ValueError as exc:
            _exit_session_error(exc)
    console.print(f"Run id: {run_id}")
    console.print(f"Status: {state.final_status}")
    if getattr(state, "final_reason", None):
        console.print(f"Reason: {state.final_reason}")
    if state.final_answer:
        console.print(state.final_answer)
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
            "enabled_tools": tool,
        },
    )
    registry = _build_registry_or_exit(loaded.enabled_tools, loaded.tool_modules)
    trace_sink = ConsoleTraceSink(console) if trace else NullTraceSink()
    session_store = SessionStore(loaded.logs_dir)

    try:
        llm = LLMClient(loaded, trace=trace_sink)
    except MissingAPIKeyError as exc:
        console.print(str(exc))
        raise typer.Exit(1) from exc

    def confirm(decision: PermissionDecision, action) -> bool:
        console.print(f"Tool requires confirmation: {action.tool_name}")
        console.print(f"Reason: {decision.reason}")
        console.print(f"Arguments: {action.arguments}")
        return typer.confirm("Allow this action?", default=False)

    console.print("Type /exit or /quit to leave.")
    while True:
        try:
            goal = typer.prompt("agent")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if goal.strip().lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if not goal.strip():
            continue

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
                    _assistant_transcript_metadata(run_id, state),
                )
            except ValueError as exc:
                _exit_session_error(exc)

        if state.final_answer:
            console.print(f"assistant> {state.final_answer}")
        else:
            console.print(f"Status: {state.final_status}")
            if getattr(state, "final_reason", None):
                console.print(f"Reason: {state.final_reason}")
        if trace:
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
    tool: Annotated[list[str] | None, typer.Option("--tool")] = None,
    trace: Annotated[bool, typer.Option("--trace")] = False,
) -> None:
    chat(
        config=config,
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        max_steps=max_steps,
        tool=tool,
        session=session,
        no_session=False,
        trace=trace,
    )


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
        console.print(json.dumps(payload, ensure_ascii=False, indent=2), soft_wrap=True)
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
            summary.name,
            str(summary.transcript_turns),
            summary.summary or "",
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
    log_path = _find_run_log(loaded.logs_dir, run_id)
    if log_path is None:
        console.print(f"Run log not found: {run_id}")
        raise typer.Exit(1)

    summary = _summarize_run_log_or_exit(
        log_path,
        run_id=run_id,
        include_events=include_events,
    )

    if as_json:
        console.print(
            json.dumps(summary, ensure_ascii=False, indent=2),
            soft_wrap=True,
        )
        return

    console.print(f"Run id: {run_id}")
    console.print(f"Status: {summary['status']}")
    if summary["reason"]:
        console.print(f"Reason: {summary['reason']}")
    if summary["goal"]:
        console.print(f"Goal: {summary['goal']}", soft_wrap=True)
    if summary["session"]:
        console.print(f"Session: {summary['session']}")
    if summary["conversation_turns"] is not None:
        console.print(f"Conversation turns: {summary['conversation_turns']}")
    if summary["steps"] is not None:
        console.print(f"Steps: {summary['steps']}")
    if summary["tool_calls"]:
        console.print(
            f"Tool calls: {summary['tool_calls']} "
            f"({summary['tool_failures']} failed)"
        )
        if summary["last_tool_error"]:
            error = summary["last_tool_error"]
            console.print(
                f"Last tool error: {error['tool_name']} - {error['summary']}",
                soft_wrap=True,
            )
    console.print(f"Log: {summary['log']}", soft_wrap=True)


@app.command(name="list-runs")
def list_runs(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1)] = 20,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    loaded = _load_config_or_exit(config)
    summaries = [
        _summarize_run_log_or_exit(log_path)
        for log_path in _run_log_paths(loaded.logs_dir)[:limit]
    ]

    if as_json:
        console.print(
            json.dumps({"runs": summaries}, ensure_ascii=False, indent=2),
            soft_wrap=True,
        )
        return

    if not summaries:
        console.print("No run logs found.")
        return

    table = Table(title="Recent Runs")
    table.add_column("Run ID")
    table.add_column("Status")
    table.add_column("Steps")
    table.add_column("Tool Failures")
    table.add_column("Goal")
    table.add_column("Log")
    for summary in summaries:
        table.add_row(
            str(summary["run_id"]),
            str(summary["status"]),
            "" if summary["steps"] is None else str(summary["steps"]),
            f"{summary['tool_failures']}/{summary['tool_calls']}",
            str(summary["goal"] or ""),
            str(summary["log"]),
        )
    console.print(table)


@app.command(name="restore-run")
def restore_run(
    run_id: str,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    session: Annotated[str, typer.Option("--session")] = "default",
) -> None:
    loaded = _load_config_or_exit(config)
    log_path = _find_run_log(loaded.logs_dir, run_id)
    if log_path is None:
        console.print(f"Run log not found: {run_id}")
        raise typer.Exit(1)

    summary = _summarize_run_log_or_exit(log_path, run_id=run_id)
    goal = summary["goal"]
    if not isinstance(goal, str) or not goal:
        console.print(f"Run log has no restorable goal: {run_id}")
        raise typer.Exit(1)

    store = SessionStore(loaded.logs_dir)
    if _session_has_restored_run(store, session, run_id):
        console.print(f"Run {run_id} is already restored in session {session}")
        return

    try:
        store.append_transcript(
            session,
            "user",
            goal,
            {"run_id": run_id, "source": "run_log"},
        )
        assistant_content = _summary_transcript_content(summary)
        if assistant_content:
            metadata = {
                "run_id": run_id,
                "source": "run_log",
                "status": summary["status"],
            }
            if summary["reason"]:
                metadata["reason"] = summary["reason"]
            store.append_transcript(session, "assistant", assistant_content, metadata)
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
    matches = sorted(
        (path for path in logs_dir.glob("*/*.jsonl") if path.name == filename),
        reverse=True,
    )
    return matches[0].resolve() if matches else None


def _run_log_paths(logs_dir: Path) -> list[Path]:
    return sorted(
        (path for path in logs_dir.glob("*/*.jsonl") if path.is_file()),
        reverse=True,
    )


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
    failed_tool_payloads = [
        event["payload"]
        for event in tool_events
        if event["payload"].get("success") is False
    ]
    last_tool_error = None
    if failed_tool_payloads:
        last_failed = failed_tool_payloads[-1]
        last_tool_error = {
            "tool_name": last_failed.get("tool_name"),
            "summary": last_failed.get("summary"),
            "error": last_failed.get("error"),
        }
    resolved_run_id = run_id
    if resolved_run_id is None and final_event and final_event.get("run_id"):
        resolved_run_id = str(final_event["run_id"])
    if resolved_run_id is None and start_event and start_event.get("run_id"):
        resolved_run_id = str(start_event["run_id"])
    if resolved_run_id is None:
        resolved_run_id = log_path.stem
    summary = {
        "run_id": resolved_run_id,
        "goal": start_payload.get("goal"),
        "workspace": start_payload.get("workspace"),
        "session": start_payload.get("session"),
        "resumed": start_payload.get("resumed"),
        "conversation_turns": start_payload.get("conversation_turns"),
        "status": final_payload.get("status", "unknown"),
        "reason": final_payload.get("reason"),
        "answer": final_payload.get("answer"),
        "steps": step,
        "tool_calls": len(tool_events),
        "tool_failures": len(failed_tool_payloads),
        "last_tool_error": last_tool_error,
        "log": str(log_path.resolve()),
    }
    if include_events:
        summary["events"] = events
    return summary


def _event_payload(event: dict[str, Any] | None) -> dict[str, Any]:
    if event is None:
        return {}
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _summary_transcript_content(summary: dict[str, object]) -> str | None:
    answer = summary.get("answer")
    if isinstance(answer, str) and answer:
        return answer

    status = summary.get("status")
    if isinstance(status, str) and status:
        content = f"Run stopped with status {status}."
        reason = summary.get("reason")
        if isinstance(reason, str) and reason:
            content = f"{content} Reason: {reason}"
        return content

    return None


def _read_run_events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        raise ValueError(f"Run log path is not a file: {log_path}")

    events: list[dict[str, Any]] = []
    try:
        raw_lines = log_path.read_bytes().splitlines()
    except OSError as exc:
        raise ValueError(f"Run log could not be read: {log_path}") from exc
    for raw_line in raw_lines:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if line.strip():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events
