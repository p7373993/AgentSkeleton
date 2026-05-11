import sys
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from agentskeleton.config import load_config
from agentskeleton.core.llm import LLMClient, MissingAPIKeyError
from agentskeleton.core.loop import AgentLoop
from agentskeleton.core.session import SessionStore
from agentskeleton.core.trace import ConsoleTraceSink, NullTraceSink
from agentskeleton.logging.run_logger import RunLogger
from agentskeleton.policy.permissions import PermissionDecision
from agentskeleton.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
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


def build_default_registry(enabled_tools: list[str] | None = None) -> ToolRegistry:
    tools = [
        ListDirTool(),
        ReadFileTool(),
        WriteFileTool(),
        ShellTool(),
        AskUserTool(),
    ]
    if enabled_tools is None:
        return ToolRegistry(tools)

    by_name = {tool.name: tool for tool in tools}
    selected = []
    for name in enabled_tools:
        try:
            selected.append(by_name[name])
        except KeyError as exc:
            raise ValueError(f"Unknown enabled tool: {name}") from exc
    return ToolRegistry(selected)


@app.command()
def tools() -> None:
    registry = build_default_registry()
    table = Table(title="Registered Tools")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Risk")

    for tool in registry.all():
        table.add_row(tool.name, tool.description, tool.risk)

    console.print(table)


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
    loaded = load_config(
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
    logger = RunLogger(loaded.logs_dir, run_id)
    registry = build_default_registry(loaded.enabled_tools)
    trace = NullTraceSink() if quiet else ConsoleTraceSink(console)
    session_store = SessionStore(loaded.logs_dir)
    conversation = []
    if not no_session:
        session_state = session_store.load(session)
        conversation = session_state.transcript

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
        session_store.append_transcript(
            session,
            "user",
            goal,
            {"run_id": run_id},
        )
    state = loop.run(
        goal,
        conversation=conversation,
        trace_context={"session": None if no_session else session},
    )
    if not no_session and state.final_answer:
        session_store.append_transcript(
            session,
            "assistant",
            state.final_answer,
            {"run_id": run_id, "status": state.final_status},
        )
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
    loaded = load_config(
        config,
        {
            "model": model,
            "base_url": base_url,
            "reasoning_effort": reasoning_effort,
            "max_steps": max_steps,
            "enabled_tools": tool,
        },
    )
    registry = build_default_registry(loaded.enabled_tools)
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
        logger = RunLogger(loaded.logs_dir, run_id)
        conversation = []
        if not no_session:
            conversation = session_store.load(session).transcript
            session_store.append_transcript(
                session,
                "user",
                goal,
                {"run_id": run_id},
            )

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
        if not no_session and state.final_answer:
            session_store.append_transcript(
                session,
                "assistant",
                state.final_answer,
                {"run_id": run_id, "status": state.final_status},
            )

        if state.final_answer:
            console.print(f"assistant> {state.final_answer}")
        else:
            console.print(f"Status: {state.final_status}")
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


def main() -> None:
    app()
