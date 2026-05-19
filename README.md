# AgentSkeleton

AgentSkeleton is a minimal local CLI agent. It runs from the current workspace,
calls registered tools, logs each run, and can keep session context between
commands.

## What It Includes

- `agent run` for one-shot agent tasks
- `agent chat` for an interactive local agent session
- `agent resume-run` for continuing from a previous run log
- `agent servers`, `agent stop-server`, and `agent restart-server` for managed
  local development servers
- Tool registry with filesystem, shell, and user-question tools
- Permission policy for read, write, shell, and interactive actions
- Run logs under `runs/`
- Session memory under `runs/sessions/`
- Eval suite coverage across filesystem, coding, data, finance, writing,
  interactive, artifact, reliability, and tool-pack scenarios
- Domain tool pack support through configured Python tool modules

## Requirements

- Python 3.12 or newer
- `uv`
- OpenAI-compatible API credentials for live model calls

Set credentials in your shell or `.env` before live runs. Common environment
variables are:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5
```

Azure-compatible deployments can use the config/environment variables supported
by the local config loader, including `AZURE_OPENAI_DEPLOYMENT`.

## Quick Start

From the repository root:

```bash
uv run agent --help
uv run agent tools
uv run agent run "hello. respond with one short sentence."
```

Run without session memory:

```bash
uv run agent run "hello" --no-session
```

Run quietly without trace output:

```bash
uv run agent run "hello" --quiet
```

Start an interactive chat:

```bash
uv run agent chat
```

Use a separate named session:

```bash
uv run agent run "summarize this workspace" --session workspace-review
uv run agent resume workspace-review
```

## Sessions

By default, `agent run` and `agent chat` use the `default` session. Previous
session turns are loaded into the next request. Older turns are compacted into a
`Prior conversation summary`, and recent turns are kept as transcript context.

Useful commands:

```bash
uv run agent sessions
uv run agent run "task" --session fresh-test
uv run agent run "task" --no-session
```

Session files are stored under:

```text
runs/sessions/<session-name>/summary.md
runs/sessions/<session-name>/transcript.jsonl
```

## Run Logs

Each run gets a JSONL log under `runs/YYYYMMDD/`.

```bash
uv run agent list-runs
uv run agent show-run <run-id>
uv run agent resume-run <run-id> "continue from that run"
```

## Managed Local Servers

The agent can start static localhost servers through the `start_static_server`
tool. Managed servers are detached from the run process, verified with an HTTP
request, and recorded under `runs/servers/`.

List, stop, and restart managed servers:

```bash
uv run agent servers
uv run agent servers --json
uv run agent stop-server <server-id>
uv run agent restart-server <server-id>
```

Each server record includes:

```text
server_id
pid
host
port
root
path
url
log_path
error_log_path
status
```

If the requested port is already busy, the server manager chooses the next
available port and verifies the final URL before reporting success.

## Validation

Run the project checks:

```bash
uv run ruff check .
uv run pytest
uv run agent eval-suite evals --json
```

Current completion baseline:

```text
ruff: all checks passed
pytest: 1020 passed, 1 skipped
eval-suite: 58 passed, 0 failed
```

## Built-In Tools

```text
list_dir    List direct children of a workspace-relative directory.
read_file   Read a UTF-8 text file inside the workspace.
write_file  Write UTF-8 text inside the workspace.
shell       Run a shell command in the configured workspace.
ask_user    Ask the user one direct question.
start_static_server  Start a managed static localhost server.
list_servers         List managed local servers.
stop_server          Stop a managed local server.
restart_server       Restart a managed local server.
```

## Configuration

The agent works without a config file by using defaults. Important defaults:

```text
model: gpt-5.5
reasoning_effort: low
max_steps: 20
permission_profile: standard
logs_dir: runs
session_context_turns: 20
session_summary_turns: 40
```

Common CLI overrides:

```bash
uv run agent run "task" --model gpt-5.4-mini
uv run agent run "task" --max-steps 5
uv run agent run "task" --tool read_file --tool list_dir
uv run agent run "task" --config path/to/config.yaml
```

Domain tool packs can be enabled by adding Python module paths to
`tool_modules` in config. The eval suite includes tool-pack scenarios that use
this extension path.
