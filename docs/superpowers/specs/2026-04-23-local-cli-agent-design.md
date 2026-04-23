# Local CLI Agent Design

Date: 2026-04-23

## Goal

Build a minimal local CLI agent that can repeatedly think, choose an action,
execute local tools, observe results, and continue until the user goal is done
or a stop condition is reached.

The first version is intentionally a skeleton. It should prove that the core
agent loop works on the local computer before adding web UI, browser control,
subagents, long-term memory, MCP, plugins, or self-improving skills.

## Product Shape

The user runs the agent from a terminal:

```bash
agent run "inspect this folder and draft a README"
agent run "run tests and fix the failure" --max-steps 30
agent tools
```

The agent may inspect files, edit files, run shell commands, and ask the user
questions. Each run writes a structured trace so failures can be debugged after
the fact.

`agent resume <run_id>` is reserved for the next milestone after the direct run
loop is proven.

## Recommended Stack

- Language: Python 3.12+
- Package manager: uv
- CLI framework: Typer
- Terminal output: Rich
- Config and schemas: Pydantic, YAML
- LLM integration: OpenAI Python SDK with the Responses API
- Testing: pytest
- Formatting and linting: ruff
- Optional later type checking: mypy

The first version should not use LangChain, LangGraph, or OpenAI Agents SDK.
Those frameworks may be useful later, but the first goal is to understand and
control the loop directly.

## Default Model Settings

Initial defaults:

```yaml
model: gpt-5.4-mini
reasoning_effort: low
max_steps: 20
```

Users can override these from CLI flags or `agent.yaml`.

Examples:

```bash
agent run "refactor this module" --model gpt-5.4 --reasoning-effort medium
agent run "summarize files" --model gpt-5.4-mini --reasoning-effort low
```

## Architecture

```text
src/agentskeleton/
  cli.py
  config.py

  core/
    loop.py
    state.py
    llm.py
    events.py

  tools/
    base.py
    registry.py
    shell.py
    filesystem.py
    user.py

  policy/
    permissions.py
    risk.py

  logging/
    run_logger.py
```

### Agent Loop

`core.loop.AgentLoop` owns the run lifecycle:

1. Build the model input from the user goal, current run state, and available
   tools.
2. Ask the LLM for the next action.
3. If the model returns final output, stop.
4. If the model requests a tool, validate and execute it.
5. Append the observation to state and logs.
6. Repeat until completion, failure, user interruption, or `max_steps`.

The loop must be deterministic outside the LLM call: tool execution, policy
checks, logging, and stop handling should be testable without live API calls.

### LLM Client

`core.llm.LLMClient` wraps OpenAI Responses API calls.

Responsibilities:

- Convert registered tools into model-callable tool schemas.
- Send the current run context.
- Parse assistant messages, function calls, and final answers into internal
  action objects.
- Preserve the response output items needed for follow-up tool calls.

The implementation should keep API-specific details in this module so the rest
of the system can stay provider-neutral enough for future experiments.

### Tool Registry

`tools.registry.ToolRegistry` maps tool names to executable tool objects.

First-version tools:

- `shell`: run a local shell command in the configured working directory.
- `read_file`: read a UTF-8 text file.
- `write_file`: write a UTF-8 text file.
- `list_dir`: list files and directories.
- `ask_user`: pause and ask the user a direct question.

Each tool exposes:

- name
- description
- JSON schema for arguments
- execute method
- risk metadata

### Permission Policy

`policy.permissions.PermissionPolicy` decides whether a tool call can run.

The first version should support three outcomes:

- allow
- confirm
- block

The policy should confirm or block risky shell operations, including:

- recursive delete
- force delete
- disk formatting
- shutdown or reboot
- credential or secret extraction
- commands outside the configured workspace when the tool call looks destructive
- network exfiltration patterns paired with sensitive paths

The policy does not need to be perfect in version one. It needs to be explicit,
testable, and conservative.

### Run Logger

`logging.run_logger.RunLogger` writes a JSONL trace per run.

Each step should log:

- run id
- step number
- timestamp
- user goal
- model selected action
- tool name and arguments
- policy decision
- tool result summary
- error details, if any
- final answer, if any

The trace should be useful for debugging failed loops without re-running the
task.

## State And Data Flow

Main flow:

```text
User Goal
  -> CLI
  -> Runtime Config
  -> AgentLoop
  -> LLMClient
  -> ToolRegistry
  -> PermissionPolicy
  -> Tool Execution
  -> RunLogger
  -> AgentLoop next step
```

Run state should contain:

- run id
- working directory
- original goal
- step count
- prior model output items needed for Responses API continuation
- tool observations
- final status

## Stop Conditions

The agent stops when one of these occurs:

- model returns a final answer
- `max_steps` is reached
- tool execution fails in a non-recoverable way
- permission policy blocks an action
- user denies a confirmation prompt
- user interrupts the process

When stopping, the CLI should print the final status, short explanation, and
path to the run log.

## Error Handling

Errors should be turned into explicit observations where possible. For example,
a shell command with a non-zero exit code should return stdout, stderr, and
exit code to the agent rather than crashing the process.

The process should crash only for internal programming errors, invalid config,
or unrecoverable setup problems such as missing API credentials.

## Configuration

The default config file is `agent.yaml`.

Example:

```yaml
model: gpt-5.4-mini
reasoning_effort: low
max_steps: 20
workspace: "."
confirm_risky_actions: true
logs_dir: "runs"
```

Environment:

```bash
OPENAI_API_KEY=...
```

## Testing Strategy

Initial tests:

- tool registry registers and looks up tools correctly
- filesystem tools read, write, and list within test temp directories
- shell tool captures stdout, stderr, exit code, and timeout
- permission policy blocks clearly dangerous commands
- permission policy allows safe read-only commands
- agent loop can run with a fake LLM client and fake tools
- run logger writes valid JSONL events

Live OpenAI tests should be optional and skipped unless an API key is present.

## Version One Exclusions

Do not include these in the first implementation:

- web UI
- browser automation
- subagents
- long-term memory or vector database
- self-improving skill storage
- MCP server or MCP client integration
- plugin system
- background daemon or scheduler
- multi-user auth
- cloud execution

## Success Criteria

Version one is successful when:

- `agent run "<goal>"` starts a local run from the terminal.
- The model can choose from registered local tools.
- The loop can execute at least shell, read, write, list, and ask-user actions.
- Risky commands are confirmed or blocked.
- Every step is logged.
- A fake-LLM smoke test proves the loop without live API calls.
- The codebase remains small enough that each module has one clear job.

## Follow-Up Milestones

After the skeleton is working:

1. Add resume support using saved run state.
2. Add richer permission profiles.
3. Add browser/computer-control tools.
4. Add compact working memory.
5. Add skill capture from successful traces.
6. Consider adopting OpenAI Agents SDK or MCP after the direct loop is proven.
