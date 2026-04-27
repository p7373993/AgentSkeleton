# Local CLI Agent Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working minimal `agent` CLI that can run a local tool-calling loop with offline fake-LLM tests and an OpenAI Responses API boundary.

**Architecture:** Keep the agent loop independent of provider SDK objects by using internal action, state, tool, policy, and event contracts. Implement filesystem and shell tools behind a registry and route side effects through a permission policy before execution. Keep OpenAI-specific request, response, and tool-call continuation code in `core.llm`.

**Tech Stack:** Python 3.12+, Typer, Rich, Pydantic, PyYAML, OpenAI Python SDK, pytest, ruff, uv.

---

### Task 1: Packaging And CLI Baseline

**Files:**
- Create: `pyproject.toml`
- Create: `agent.yaml.example`
- Create: `src/agentskeleton/__init__.py`
- Create: `src/agentskeleton/cli.py`
- Create: `src/agentskeleton/config.py`
- Create: `tests/test_config.py`
- Create: `tests/test_cli.py`

- [ ] Write failing tests for config defaults and CLI tool listing.
- [ ] Run targeted tests and verify they fail because modules do not exist.
- [ ] Add package metadata, dependencies, config loading, and a Typer CLI with `run` and `tools` commands.
- [ ] Run targeted tests and verify they pass.

### Task 2: Core Contracts And Registry

**Files:**
- Create: `src/agentskeleton/core/actions.py`
- Create: `src/agentskeleton/core/events.py`
- Create: `src/agentskeleton/core/state.py`
- Create: `src/agentskeleton/tools/base.py`
- Create: `src/agentskeleton/tools/registry.py`
- Create: `tests/test_tool_registry.py`

- [ ] Write failing tests for tool registration, lookup, duplicate rejection, and schema export.
- [ ] Run targeted tests and verify they fail.
- [ ] Add Pydantic/dataclass contracts and registry implementation.
- [ ] Run targeted tests and verify they pass.

### Task 3: Path Resolver And Filesystem Tools

**Files:**
- Create: `src/agentskeleton/policy/paths.py`
- Create: `src/agentskeleton/tools/filesystem.py`
- Create: `tests/test_filesystem_tools.py`

- [ ] Write failing tests for workspace-relative path resolution, escape blocking, listing, reading, writing, and binary rejection.
- [ ] Run targeted tests and verify they fail.
- [ ] Implement the resolver and filesystem tools.
- [ ] Run targeted tests and verify they pass.

### Task 4: Permission Policy And Shell Tool

**Files:**
- Create: `src/agentskeleton/policy/risk.py`
- Create: `src/agentskeleton/policy/permissions.py`
- Create: `src/agentskeleton/tools/shell.py`
- Create: `tests/test_permission_policy.py`
- Create: `tests/test_shell_tool.py`

- [ ] Write failing tests for allow/confirm/block decisions and shell stdout, stderr, exit code, timeout, and truncation behavior.
- [ ] Run targeted tests and verify they fail.
- [ ] Implement the policy and shell tool.
- [ ] Run targeted tests and verify they pass.

### Task 5: Run Logger

**Files:**
- Create: `src/agentskeleton/logging/run_logger.py`
- Create: `tests/test_run_logger.py`

- [ ] Write failing tests for JSONL event writing and secret redaction.
- [ ] Run targeted tests and verify they fail.
- [ ] Implement the run logger.
- [ ] Run targeted tests and verify they pass.

### Task 6: Agent Loop With Fake LLM

**Files:**
- Create: `src/agentskeleton/core/loop.py`
- Create: `tests/test_agent_loop.py`
- Modify: `src/agentskeleton/cli.py`

- [ ] Write failing tests for final-answer completion, tool-call execution, max-step stop, and blocked-action stop.
- [ ] Run targeted tests and verify they fail.
- [ ] Implement `AgentLoop` with injectable LLM, registry, policy, logger, and confirmer dependencies.
- [ ] Wire the CLI to build default tools and run the loop.
- [ ] Run targeted tests and verify they pass.

### Task 7: OpenAI LLM Boundary

**Files:**
- Create: `src/agentskeleton/core/llm.py`
- Create: `tests/test_llm.py`
- Modify: `src/agentskeleton/cli.py`

- [ ] Write failing tests with fake OpenAI SDK objects for tool schema conversion, final-text parsing, function-call parsing, and missing API key errors.
- [ ] Run targeted tests and verify they fail.
- [ ] Implement `LLMClient` without requiring a live API call in unit tests.
- [ ] Run targeted tests and verify they pass.

### Task 8: Full Verification

**Files:**
- Modify: implementation files only if verification exposes defects.

- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run agent tools`.
- [ ] Run an offline fake-loop smoke test if exposed by tests.
- [ ] Report exact verification results and remaining gaps.
