# Local CLI Agent Design

Date: 2026-04-23
Revised: 2026-04-27

## Goal

Build a small local CLI agent that can run on one developer machine, repeatedly
ask a model for the next action, execute local tools, feed observations back to
the model, and stop with an explicit result.

Version one is intentionally narrow. It should prove the local agent loop,
tool-call protocol, permission policy, and run logging before adding a web UI,
browser automation, subagents, long-term memory, MCP, plugins, background
workers, or hosted execution.

## Design Principles

- Keep the loop understandable. Each step should be visible in code and logs.
- Keep side effects explicit. File writes and shell commands pass through one
  policy layer before they execute.
- Keep API details isolated. OpenAI-specific request and response handling stays
  inside the LLM client.
- Keep tools boring. Tool argument schemas should be shallow, typed, and easy
  for the model to fill correctly.
- Keep tests offline by default. A fake LLM must prove the loop without a live
  API key.

## Product Shape

The user runs the agent from a terminal:

```bash
agent run "inspect this folder and draft a README"
agent run "run tests and fix the failure" --max-steps 30
agent tools
```

The agent may inspect files, edit files, run shell commands, and ask the user a
direct question. Every run writes a JSONL trace that explains what happened
without requiring the task to be rerun.

`agent resume <run_id>` is reserved for the next milestone. Version one may log
enough state to make resume possible later, but it does not need to implement
resume behavior.

## Version One Scope

Include:

- `agent run "<goal>"` for a single local run.
- `agent tools` to list registered tools, descriptions, and risk levels.
- Workspace-scoped filesystem tools.
- A shell tool with timeout, output limits, and policy checks.
- An interactive `ask_user` tool.
- JSONL run logs.
- Offline tests with fake model and fake tools.
- Optional live OpenAI smoke tests gated by `OPENAI_API_KEY`.

Exclude:

- Web UI.
- Browser or computer-control tools.
- Subagents.
- Long-term memory or vector search.
- MCP server or MCP client integration.
- Plugin system.
- Background daemon, scheduler, or queue.
- Multi-user authentication.
- Cloud or remote execution.

## Recommended Stack

- Language: Python 3.12+
- Package manager: `uv`
- CLI framework: Typer
- Terminal output: Rich
- Config and schemas: Pydantic and YAML
- LLM integration: OpenAI Python SDK with the Responses API
- Testing: pytest
- Formatting and linting: ruff
- Optional later type checking: mypy

Do not use LangChain, LangGraph, or the OpenAI Agents SDK in version one. Those
may be useful after the direct loop is proven, but the first implementation
should make orchestration and state handling explicit.

## Default Model Settings

Use config keys that are stable for this project, then map them to the OpenAI
Responses API inside `core.llm`.

Initial defaults:

```yaml
model: gpt-5.5
reasoning_effort: low
text_verbosity: low
max_steps: 20
model_retry_attempts: 2
```

The current OpenAI guidance says GPT-5.5 works best through the Responses API,
supports tool-heavy workflows, uses `reasoning.effort`, and can continue state
with `previous_response_id`. This design should not scatter those assumptions
through the codebase. Keep them in `LLMClient` and make model settings
overridable through CLI flags or `agent.yaml`. Use `gpt-5.4-mini` when latency
and cost matter more than peak capability.

Examples:

```bash
agent run "refactor this module" --reasoning-effort medium
agent run "summarize files" --model gpt-5.4-mini --reasoning-effort low
```

## Project Layout

```text
pyproject.toml
agent.yaml.example
src/agentskeleton/
  __init__.py
  cli.py
  config.py

  core/
    actions.py
    events.py
    llm.py
    loop.py
    state.py

  tools/
    base.py
    filesystem.py
    registry.py
    shell.py
    user.py

  policy/
    permissions.py
    paths.py
    risk.py

  logging/
    run_logger.py

tests/
  test_agent_loop.py
  test_config.py
  test_filesystem_tools.py
  test_permission_policy.py
  test_run_logger.py
  test_shell_tool.py
  test_tool_registry.py
```

The split is intentionally small. If an implementation needs more modules, add
them only when a file has more than one clear responsibility.

## Core Data Contracts

Define internal contracts before wiring real API calls. The loop should depend
on these contracts, not on raw OpenAI response objects.

- `RunConfig`: model, reasoning effort, max steps, workspace, log directory,
  timeout defaults, output limits, and confirmation behavior.
- `RunState`: run id, original goal, workspace root, step count, prior response
  id, prior response output items if needed, observations, and final status.
- `AgentAction`: discriminated union of `ToolCallAction` and `FinalAction`.
- `ToolCallAction`: tool name, validated arguments, provider call id, and raw
  provider metadata needed to continue the response.
- `FinalAction`: final text and optional structured status.
- `ToolResult`: success flag, structured payload, human-readable summary, and
  optional error details.
- `ToolObservation`: tool call id, tool name, policy decision, result summary,
  and full result payload for model continuation.
- `RunEvent`: append-only log event with a type, timestamp, run id, step number,
  and event-specific payload.

These contracts should be Pydantic models or dataclasses with focused tests.

## Runtime Flow

```text
User Goal
  -> CLI
  -> RunConfig
  -> AgentLoop
  -> LLMClient
  -> ToolRegistry
  -> PermissionPolicy
  -> Tool Execution
  -> RunLogger
  -> AgentLoop next step
```

`AgentLoop` owns the lifecycle:

1. Create a run id and initialize state.
2. Build model input from the original goal, available tools, and current
   observations.
3. Ask `LLMClient` for the next action.
4. If the model returns a final answer, log it and stop.
5. If the model requests a tool, validate the tool name and arguments.
6. Ask `PermissionPolicy` for `allow`, `confirm`, or `block`.
7. If confirmation is required, ask the user before execution.
8. Execute the tool or record the denial/block as an observation.
9. Send the observation back into the next loop step.
10. Stop on final answer, max steps, blocked action, denied confirmation,
    unrecoverable internal error, or user interruption.

The loop must be deterministic outside the model call. Tool execution, policy
checks, logging, and stop handling should be testable with fake dependencies.

## OpenAI Responses API Boundary

`core.llm.LLMClient` is the only module that knows the Responses API shape.

Responsibilities:

- Convert registered local tools into OpenAI function tool definitions.
- Send model, reasoning effort, verbosity, instructions, current input, and tool
  definitions.
- Parse model output into `ToolCallAction` or `FinalAction`.
- Preserve `previous_response_id` when using stateful continuation.
- Preserve returned output items for stateless or future Zero Data Retention
  flows.
- Return tool call outputs with the matching provider `call_id`.
- Hide SDK object shapes from the rest of the codebase.

The implementation should prefer `previous_response_id` for version one because
it keeps the local loop simpler. It should still store enough response metadata
in `RunState` and logs to support manual output-item replay later.

## Tool System

`tools.registry.ToolRegistry` maps tool names to executable tool objects.

Each tool exposes:

- `name`
- `description`
- JSON schema for arguments
- risk metadata
- `execute(args, context) -> ToolResult`

Tool schemas should avoid deep nesting unless a tool genuinely needs it.
Descriptions should state what the tool does, when to use it, side effects,
input constraints, retry safety, and common error modes.

Version-one tools:

- `list_dir`: list direct children of a workspace-relative directory.
- `read_file`: read a UTF-8 text file inside the workspace.
- `write_file`: write UTF-8 text inside the workspace, creating parent
  directories only when explicitly requested.
- `shell`: run a command in the configured workspace.
- `ask_user`: pause the loop and ask the user one direct question.

## Filesystem Rules

All filesystem tools accept workspace-relative paths. A shared path resolver in
`policy.paths` must:

- Resolve paths against the configured workspace root.
- Normalize `.` and `..`.
- Reject paths that escape the workspace.
- Reject symlink escapes.
- Return stable absolute paths to tools.

`read_file` should reject binary files and return a clear error observation.
`write_file` should write atomically where practical, return the byte count, and
avoid changing permissions. Version one does not need patch-based editing.

## Shell Tool

The shell tool runs one command in the configured workspace and returns:

- command
- working directory
- exit code
- stdout
- stderr
- duration
- timeout flag
- truncation flags

Defaults:

```yaml
shell_timeout_seconds: 30
shell_max_output_bytes: 20000
```

Non-zero exit codes are tool results, not internal crashes. Timeouts should
terminate the process and return a timeout observation. Output truncation should
be visible to both the model and the run log.

Version one may pass the inherited environment, but logs must never record full
environment variables. If environment capture is later needed, use an allowlist.

## Permission Policy

`policy.permissions.PermissionPolicy` decides whether a tool call can run.

Outcomes:

- `allow`: execute immediately.
- `confirm`: ask the user before executing.
- `block`: do not execute; return a blocked observation.

Default behavior:

- Allow read-only filesystem tools inside the workspace.
- Confirm `write_file` unless `confirm_risky_actions` is disabled.
- Confirm shell commands that modify files, install packages, access the
  network, spawn long-running processes, or run outside common read-only
  commands.
- Block clearly dangerous commands.

The policy should block or require confirmation for:

- recursive delete
- force delete
- disk formatting
- shutdown or reboot
- credential or secret extraction
- destructive commands targeting paths outside the workspace
- network exfiltration patterns paired with sensitive paths
- shell metacharacter chains that hide destructive follow-up commands

The policy does not need to be perfect. It must be explicit, conservative, and
covered by table-driven tests.

## User Interaction

The CLI should print concise progress:

- run id
- current step
- selected tool
- policy decision
- final status
- run log path

For `confirm`, show the exact tool, arguments, risk reason, and expected side
effect. The user can approve or deny. Denial becomes a normal observation so
the model can either choose another path or stop.

For `ask_user`, the model supplies a direct question. The CLI displays it and
captures a single response. The response is logged as an observation.

## Run Logger

`logging.run_logger.RunLogger` writes one JSON object per line.

Log path:

```text
runs/<YYYYMMDD>/<run_id>.jsonl
```

Event types:

- `run_started`
- `model_requested`
- `model_action`
- `policy_decision`
- `tool_started`
- `tool_finished`
- `user_confirmation`
- `user_answered`
- `run_finished`
- `run_error`

Each event includes run id, step number, timestamp, and a payload. Tool outputs
may be truncated in the log, but truncation must be marked. Secrets should be
redacted with a small set of conservative patterns for API keys, bearer tokens,
and common credential names.

The trace should be useful for debugging failed loops without rerunning the
task.

## Configuration

Default config file: `agent.yaml`.

Config precedence:

1. CLI flags.
2. `agent.yaml`.
3. built-in defaults.

Example:

```yaml
model: gpt-5.5
reasoning_effort: low
text_verbosity: low
max_steps: 20
model_retry_attempts: 2
workspace: "."
confirm_risky_actions: true
logs_dir: "runs"
shell_timeout_seconds: 30
shell_max_output_bytes: 20000
```

Environment:

```bash
OPENAI_API_KEY=...
```

The app should fail fast with a clear setup error when a live LLM run is
requested without an API key. Offline tests and fake-LLM runs must not require
one.

## Error Handling

Expected failures become observations:

- file not found
- invalid path
- binary file rejected
- permission denied by policy
- user denied confirmation
- shell non-zero exit
- shell timeout
- malformed tool arguments
- unknown tool name

Internal programming errors may crash the process after logging `run_error`.
Invalid config should stop before a run starts.

The final CLI output should always include:

- final status
- short explanation
- run id
- path to the run log

## Testing Strategy

Offline tests are required:

- Config precedence and validation.
- Tool registry registration and lookup.
- Workspace path resolver blocks escape attempts and symlink escapes.
- Filesystem tools read, write, list, and reject invalid inputs.
- Shell tool captures stdout, stderr, exit code, timeout, and truncation.
- Permission policy allows safe read-only commands.
- Permission policy confirms or blocks risky commands.
- Run logger writes valid JSONL and redacts obvious secrets.
- Agent loop completes with a fake LLM final answer.
- Agent loop executes a fake tool call and feeds the observation back.
- Agent loop stops on max steps, blocked action, and denied confirmation.

Optional live tests:

- A live Responses API smoke test runs only when `OPENAI_API_KEY` is present and
  an explicit marker or environment flag enables it.

## Implementation Order

1. Create packaging, config, and CLI shell.
2. Define core contracts and fake LLM interfaces.
3. Implement tool base classes and registry.
4. Implement path resolver and filesystem tools.
5. Implement permission policy.
6. Implement shell tool.
7. Implement run logger.
8. Implement `AgentLoop` against fake LLM and fake tools.
9. Implement OpenAI `LLMClient`.
10. Add optional live smoke test and documentation.

This order keeps the loop testable before any live model dependency exists.

## Success Criteria

Version one is successful when:

- `agent run "<goal>"` starts a local run from the terminal.
- The model can choose from registered local function tools.
- The loop can execute shell, read, write, list, and ask-user actions.
- Filesystem tools cannot escape the configured workspace.
- Risky commands are confirmed or blocked.
- Every step is logged as JSONL.
- A fake-LLM smoke test proves the loop without live API calls.
- The codebase remains small enough that each module has one clear job.

## Follow-Up Milestones

After the skeleton is working:

1. Add resume support using saved run state.
2. Add richer permission profiles.
3. Add diff or patch-based editing.
4. Add browser or computer-control tools.
5. Add compact working memory.
6. Add skill capture from successful traces.
7. Consider OpenAI Agents SDK or MCP after the direct loop is proven.

## Source Notes

OpenAI API assumptions in this document were checked against the official
OpenAI developer docs on 2026-04-27:

- `https://developers.openai.com/api/docs/guides/latest-model`
- `https://developers.openai.com/api/docs/guides/function-calling`
- `https://developers.openai.com/api/docs/models`
