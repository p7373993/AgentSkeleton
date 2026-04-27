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


def build_default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ListDirTool(),
            ReadFileTool(),
            WriteFileTool(),
            ShellTool(),
            AskUserTool(),
        ]
    )


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
        },
    )
    run_id = str(uuid4())
    logger = RunLogger(loaded.logs_dir, run_id)
    registry = build_default_registry()
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
    console.print(f"Status: {state.final_status}")
    if state.final_answer:
        console.print(state.final_answer)
    console.print(f"Run log: {logger.path}")


def main() -> None:
    app()
