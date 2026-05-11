# Agent Loop Resilience Design

Date: 2026-05-11
Status: Draft for review

## Goal

Strengthen the local CLI agent loop so routine model, tool, and orchestration
failures become observable run state instead of fragile process behavior. This
is the first milestone toward a smoother agent system that can later support
broader domains, richer tools, and long-running workflows.

## Current Baseline

The repository already includes a small local agent skeleton:

- `agent run` starts a single local run.
- `agent tools` lists registered tools.
- The default registry exposes `list_dir`, `read_file`, `write_file`, `shell`,
  and `ask_user`.
- `AgentLoop` owns model calls, policy checks, tool execution, observations,
  and final status.
- `LLMClient` isolates the OpenAI Responses API boundary.
- `SessionStore` persists chat-style transcript turns.
- Offline tests cover the existing loop, tools, policy, logger, LLM boundary,
  trace output, and CLI behavior.

Recent uncommitted changes add an `agent chat` command and matching CLI tests.
This design does not depend on reverting or restructuring those changes.

## Problem

The current loop handles happy-path completion, max-step exhaustion, policy
blocks, and denied confirmations. It is less explicit about common agent-loop
failure modes:

- A model can request an unknown tool and the loop records a generic error.
- A tool can raise an exception and crash the run instead of producing a normal
  observation.
- A model can repeat the same unproductive tool action until `max_steps`.
- Final statuses do not always make the failure reason precise enough for
  debugging, resume support, or future policy decisions.

These gaps matter before adding more domains. More tools and longer tasks
increase the chance of malformed, repeated, or failing actions.

## Design Principles

- Keep failure observable. Expected operational failures should become
  `ToolObservation` records and run log events.
- Keep the loop deterministic outside model calls. Classification, fingerprints,
  and stop decisions should be testable with fake LLMs and fake tools.
- Keep the first milestone small. Do not add subagents, plugins, MCP, browser
  control, vector memory, or hosted execution in this change.
- Preserve existing contracts unless a small extension directly supports loop
  resilience.

## Scope

Include:

- Distinct final statuses for unknown tools, tool execution errors, and repeated
  actions.
- A stable action fingerprint for detecting repeated tool calls.
- A configurable repeated-action threshold stored in code for this milestone.
- Conversion of tool exceptions into failed `ToolObservation` entries.
- Run log and trace events that make the stop reason visible.
- Offline tests for every new failure mode.

Exclude:

- Persistent run resume.
- Domain-specific tool packs.
- Plugin loading.
- Prompt memory compression.
- Parallel or speculative tool execution.
- Live OpenAI integration tests.

## Proposed Statuses

Keep the existing statuses:

- `completed`
- `max_steps`
- `blocked`
- `denied`

Add explicit statuses:

- `unknown_tool`: the model requested a tool that is not registered.
- `tool_error`: a registered tool raised an unexpected exception.
- `repeated_action`: the same tool name and arguments repeated enough times to
  indicate a stuck loop.

The loop should continue to store human-readable failure details in the
observation result and the `run_finished` event payload.

## Action Fingerprint

Add a small deterministic fingerprint for `ToolCallAction`:

```text
tool_name + canonical JSON arguments
```

Rules:

- Sort JSON keys.
- Use compact separators.
- Convert non-JSON values with `str()` only if needed.
- Include the tool name so two tools with the same arguments do not collide.

`RunState` should track:

- `last_action_fingerprint: str | None`
- `repeated_action_count: int`

On each tool call:

1. Compute the fingerprint.
2. If it matches the previous fingerprint, increment the repeat count.
3. If it differs, replace the previous fingerprint and reset the repeat count
   to 1.
4. Stop with `repeated_action` when the count reaches the threshold.

For this milestone, the threshold should be 3. That catches obvious loops while
still allowing a model to retry a transient tool result twice.

## Unknown Tool Handling

When the registry lookup fails:

1. Create a failed `ToolResult` with:
   - `summary`: the registry error text.
   - `error`: `Unknown tool`.
   - `payload`: the requested tool name and arguments.
2. Append a `ToolObservation` with policy decision `block`.
3. Set final status to `unknown_tool`.
4. Emit `run_finished` with the status and failure summary.

This should replace the current generic `error` final status for unknown tools.

## Tool Exception Handling

When `tool.execute()` raises:

1. Catch the exception in `AgentLoop`.
2. Create a failed `ToolResult` with:
   - `summary`: `Tool raised an exception: <ExceptionClass>`.
   - `error`: the exception message.
   - `payload`: tool name and arguments.
3. Append a `ToolObservation` using the already computed policy decision.
4. Log `tool_finished` with `success=false`.
5. Set final status to `tool_error`.
6. Emit `run_finished` with the status and failure summary.

This keeps tool bugs debuggable without making the CLI process terminate before
the run log is closed.

## Logging And Trace

The existing event stream should remain append-only. The resilience changes
should reuse existing event types where possible:

- `model_action`: include the tool call and arguments.
- `policy_decision`: include the decision when a registered tool exists.
- `tool_finished`: include failed tool exception results.
- `run_finished`: include `status`, `answer`, and a new optional `reason`.

Unknown tools do not have a policy decision from a real tool, so they should
log `model_action` and then finish with `unknown_tool`.

## Testing Strategy

Add focused offline tests to `tests/test_agent_loop.py`:

- Unknown tool requests stop with `unknown_tool` and append a failed
  observation.
- Tool exceptions stop with `tool_error`, append a failed observation, and log
  `tool_finished`.
- Repeating the same tool call three times stops with `repeated_action`.
- A different tool argument resets the repeat counter.
- Existing final answer, tool execution, max steps, blocked, denied, ask-user,
  and conversation-history tests continue to pass.

Run full verification:

```bash
uv run pytest
uv run ruff check .
```

## Success Criteria

This milestone is complete when:

- Every new failure mode has a failing test first and then a passing
  implementation.
- `AgentLoop` returns precise final statuses for unknown tools, tool exceptions,
  and repeated actions.
- The run state contains enough information to explain why the loop stopped.
- Existing CLI, LLM, tool, policy, logging, session, and trace tests still pass.
- No unrelated refactor or domain expansion is included.

## Prompt-To-Artifact Checklist

The active thread goal is broader than this milestone. This checklist separates
what this design covers from what must remain follow-up work.

| Goal phrase | Covered by this milestone | Evidence required |
| --- | --- | --- |
| Smooth agent loop | Yes | `tests/test_agent_loop.py` includes unknown-tool, tool-exception, repeated-action, and reset-counter tests. |
| Agent system development | Partially | Core loop state, observations, logs, and statuses become stronger; no new domain systems are added. |
| Works across domains | Indirectly | The loop becomes safer for future domain tool packs, but domain-specific tools are out of scope. |
| Stable operation | Partially | Expected operational failures become explicit final statuses and log events; persistent resume is out of scope. |

## Implementation Readiness Gate

Implementation should start only after review of this design. The first
implementation plan should be limited to the files below:

- `src/agentskeleton/core/state.py`
- `src/agentskeleton/core/loop.py`
- `tests/test_agent_loop.py`

The plan should not modify CLI command behavior, LLM request shaping, shell
execution, filesystem tools, session storage, or documentation outside this
milestone unless a failing test exposes a direct dependency.

## Follow-Up Milestones

After this milestone:

1. Add persistent run snapshots and `agent resume`.
2. Add observation summarization for long sessions.
3. Add domain tool-pack loading behind the existing registry.
4. Add richer policy profiles for different workspace trust levels.
5. Add browser or computer-control tools after the loop remains stable under
   failing and repeated actions.
