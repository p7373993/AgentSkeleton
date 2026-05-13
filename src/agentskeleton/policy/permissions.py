import re
from dataclasses import dataclass
from typing import Any, Literal

PermissionOutcome = Literal["allow", "confirm", "block"]
PermissionProfile = Literal["standard", "read_only", "trusted"]
FETCH_COMMANDS = {
    "curl",
    "wget",
    "invoke-webrequest",
    "invoke-restmethod",
    "iwr",
    "irm",
}
EXPRESSION_EXECUTOR_COMMANDS = {"iex", "invoke-expression"}
GIT_NETWORK_SUBCOMMANDS = {"clone", "fetch", "ls-remote", "pull", "push"}
GH_NETWORK_SUBCOMMANDS = {
    "api",
    "attestation",
    "auth",
    "cache",
    "codespace",
    "extension",
    "gist",
    "issue",
    "label",
    "pr",
    "project",
    "release",
    "repo",
    "ruleset",
    "run",
    "search",
    "secret",
    "ssh-key",
    "status",
    "variable",
    "workflow",
}
POWERSHELL_COMMANDS = {"powershell", "pwsh"}
SHELL_EVAL_COMMANDS = {
    "bash",
    "node",
    "perl",
    "powershell",
    "pwsh",
    "python",
    "python3",
    "ruby",
    "sh",
    "zsh",
}
PYTHON_NETWORK_IMPORT_RE = re.compile(
    r"\bfrom\s+(requests|urllib(?:\.[a-z0-9_]+)*|http\.client|socket)\b"
    r"|\bimport\s+[^;\n\r]*\b"
    r"(requests|urllib(?:\.[a-z0-9_]+)*|http\.client|socket)\b"
)
PYTHON_DYNAMIC_NETWORK_IMPORT_RE = re.compile(
    r"\b__import__\s*\(\s*['\"]"
    r"(requests|urllib(?:\.[a-z0-9_]+)*|http\.client|socket)['\"]"
    r"|\bimportlib\.import_module\s*\(\s*['\"]"
    r"(requests|urllib(?:\.[a-z0-9_]+)*|http\.client|socket)['\"]"
)
PYTHON_DYNAMIC_EXEC_RE = re.compile(r"\b(exec|eval)\s*\(")


@dataclass(frozen=True)
class PermissionDecision:
    outcome: PermissionOutcome
    reason: str


class PermissionPolicy:
    def __init__(
        self,
        confirm_risky_actions: bool = True,
        profile: PermissionProfile = "standard",
    ) -> None:
        self.confirm_risky_actions = confirm_risky_actions
        self.profile = profile

    def decide(
        self,
        tool_name: str,
        args: dict[str, Any],
        risk: str,
    ) -> PermissionDecision:
        if tool_name == "write_file" or risk == "write":
            if self.profile == "read_only":
                return PermissionDecision(
                    "block",
                    "Tool is blocked by read-only profile",
                )
            if self.confirm_risky_actions and self.profile != "trusted":
                return PermissionDecision(
                    "confirm",
                    "Tool writes files in the workspace",
                )
            return PermissionDecision("allow", "file writes allowed by configuration")

        if tool_name == "shell" or risk == "shell":
            if self.profile == "read_only":
                return PermissionDecision(
                    "block",
                    "Tool is blocked by read-only profile",
                )
            if not isinstance(args, dict):
                return PermissionDecision("block", "Shell command invalid")
            command = args.get("command")
            if not isinstance(command, str) or not _safe_strip(command):
                return PermissionDecision("block", "Shell command invalid")
            return self._decide_shell(command)

        if risk == "read" or risk == "interactive":
            return PermissionDecision("allow", "read-only or interactive tool")

        if self.profile == "read_only":
            return PermissionDecision(
                "block",
                "Tool is blocked by read-only profile",
            )

        return PermissionDecision(
            "confirm",
            f"Unknown risk level: {_safe_text(risk)}",
        )

    def _decide_shell(self, command: str) -> PermissionDecision:
        normalized = _safe_lower(command)
        if normalized is None:
            return PermissionDecision("block", "Shell command invalid")
        destructive_patterns = [
            "rm -rf /",
            "rm -rf *",
            "rmdir /s",
            "del /f /s",
            "format ",
            "shutdown",
            "reboot",
        ]
        if any(pattern in normalized for pattern in destructive_patterns):
            return PermissionDecision("block", "Shell command looks destructive")

        if _looks_like_encoded_command(normalized):
            return PermissionDecision("block", "Shell command uses encoded content")

        if _looks_like_remote_execution(normalized):
            return PermissionDecision("block", "Shell command executes remote content")

        if _looks_like_recursive_delete(normalized):
            return PermissionDecision("block", "Shell command looks destructive")

        secret_patterns = [
            ".env",
            "id_rsa",
            "authorization:",
            "bearer ",
            "sk-",
            "api_key",
            "apikey",
            "access_key",
            "credential",
            "password",
            "passwd",
            "private_key",
            "secret",
            "token",
        ]
        network_library_patterns = [
            "requests.",
            "urllib.",
            "http.client",
            "socket.",
        ]
        has_network_access = (
            _contains_fetch_command(normalized)
            or _contains_git_network_command(normalized)
            or _contains_gh_network_command(normalized)
            or _contains_python_network_import(normalized)
        ) or any(
            net in normalized for net in network_library_patterns
        )
        if has_network_access and any(
            secret in normalized for secret in secret_patterns
        ):
            return PermissionDecision(
                "block",
                "Shell command may exfiltrate credentials",
            )

        confirm_patterns = [
            " install ",
            "pip install",
            "uv pip install",
            "uv sync",
            "uv add",
            "npm install",
            "npm ci",
            "pnpm install",
            "yarn add",
            "poetry install",
            "git push",
            "start ",
            "start-process",
            "nohup ",
            "python -m http.server",
            "npm run dev",
            "uvicorn ",
        ]
        if (
            self.confirm_risky_actions
            and self.profile != "trusted"
            and (
                has_network_access
                or any(pattern in normalized for pattern in confirm_patterns)
            )
        ):
            return PermissionDecision(
                "confirm",
                "Shell command may install, network, or spawn",
            )

        return PermissionDecision("allow", "shell command allowed")


def _safe_text(value: object) -> str:
    try:
        return str(value)
    except Exception:
        return "<uninspectable>"


def _safe_strip(value: str) -> str | None:
    try:
        return value.strip()
    except Exception:
        return None


def _safe_lower(value: str) -> str | None:
    try:
        return value.lower()
    except Exception:
        return None


def _looks_like_recursive_delete(command: str) -> bool:
    delete_commands = ("remove-item", "rm", "ri", "del", "erase", "rd", "rmdir")
    tokens = command.replace(";", " ").replace("|", " ").split()
    if not any(
        _normalize_executable_token(token) in delete_commands for token in tokens
    ):
        return False
    return any(
        token in {"-recurse", "-recursive", "--recursive", "-r", "/s"}
        or _is_compact_recursive_flag(token)
        for token in tokens
    )


def _is_compact_recursive_flag(token: str) -> bool:
    return token.startswith("-") and len(token) <= 4 and "r" in token[1:]


def _looks_like_encoded_command(command: str) -> bool:
    segments = _split_shell_segments(command)
    encoded_flags = {"-encodedcommand", "-encoded", "-enco", "-enc", "-ec", "-e"}
    for segment in segments:
        tokens = segment.split()
        if _first_executable_token(tokens) not in POWERSHELL_COMMANDS:
            continue
        if any(_normalize_option_token(token) in encoded_flags for token in tokens):
            return True
    return False


def _looks_like_remote_execution(command: str) -> bool:
    executor_commands = (
        "bash",
        "cmd",
        "sh",
        "zsh",
        "powershell",
        "pwsh",
        "iex",
        "invoke-expression",
        "python",
        "python3",
        "node",
    )
    segments = _split_shell_segments(command)
    if _looks_like_python_download_then_execute(command):
        return True
    if any(_same_segment_executes_fetched_content(segment) for segment in segments):
        return True
    for index, segment in enumerate(segments[:-1]):
        if not _segment_has_fetch_command(segment):
            continue
        for next_segment in segments[index + 1 :]:
            next_tokens = next_segment.split()
            executable = _first_executable_token(next_tokens)
            if executable and (
                executable in executor_commands
                or _looks_like_script_executable(executable)
            ):
                return True
    return False


def _looks_like_python_download_then_execute(command: str) -> bool:
    return (
        _contains_python_network_import(command)
        and PYTHON_DYNAMIC_EXEC_RE.search(command) is not None
    )


def _same_segment_executes_fetched_content(segment: str) -> bool:
    if not _segment_has_fetch_command(segment):
        return False
    tokens = [_normalize_executable_token(token) for token in segment.split()]
    if any(token in EXPRESSION_EXECUTOR_COMMANDS for token in tokens):
        return True
    executable = _first_executable_token(segment.split())
    return executable in SHELL_EVAL_COMMANDS and (
        "$(" in segment or "`" in segment
    )


def _contains_fetch_command(command: str) -> bool:
    return any(
        _segment_has_fetch_command(segment)
        for segment in _split_shell_segments(command)
    )


def _contains_git_network_command(command: str) -> bool:
    return any(
        _segment_has_git_network_command(segment)
        for segment in _split_shell_segments(command)
    )


def _contains_gh_network_command(command: str) -> bool:
    return any(
        _segment_has_gh_network_command(segment)
        for segment in _split_shell_segments(command)
    )


def _contains_python_network_import(command: str) -> bool:
    return _python_command_contains_dynamic_network_import(command) or any(
        _segment_has_python_network_import(segment)
        for segment in _split_shell_segments(command)
    )


def _python_command_contains_dynamic_network_import(command: str) -> bool:
    if not _contains_python_command(command):
        return False
    return PYTHON_DYNAMIC_NETWORK_IMPORT_RE.search(command) is not None


def _contains_python_command(command: str) -> bool:
    return any(
        _first_executable_token(segment.split()) in {"python", "python3"}
        for segment in _split_shell_segments(command)
    )


def _segment_has_fetch_command(segment: str) -> bool:
    return any(
        _normalize_executable_token(token) in FETCH_COMMANDS
        for token in segment.split()
    )


def _segment_has_git_network_command(segment: str) -> bool:
    tokens = segment.split()
    if _first_executable_token(tokens) != "git":
        return False
    return any(
        _normalize_option_token(token) in GIT_NETWORK_SUBCOMMANDS
        for token in tokens[1:]
    )


def _segment_has_gh_network_command(segment: str) -> bool:
    tokens = segment.split()
    if _first_executable_token(tokens) != "gh":
        return False
    return any(
        _normalize_option_token(token) in GH_NETWORK_SUBCOMMANDS
        for token in tokens[1:]
    )


def _segment_has_python_network_import(segment: str) -> bool:
    tokens = segment.split()
    if _first_executable_token(tokens) not in {"python", "python3"}:
        return False
    return PYTHON_NETWORK_IMPORT_RE.search(segment) is not None


def _first_executable_token(tokens: list[str]) -> str | None:
    wrappers = {"sudo", "env"}
    for token in tokens:
        if token in wrappers:
            continue
        if _looks_like_env_assignment(token):
            continue
        return _normalize_executable_token(token)
    return None


def _looks_like_env_assignment(token: str) -> bool:
    key, separator, _value = token.partition("=")
    return bool(separator and key and key.replace("_", "").isalnum())


def _normalize_executable_token(token: str) -> str:
    executable = (
        token.strip("\"'`")
        .lstrip("$")
        .strip("(){}[];,")
        .lstrip("&.")
        .rsplit("/", 1)[-1]
        .rsplit("\\", 1)[-1]
    )
    if executable.endswith(".exe"):
        executable = executable[:-4]
    return executable


def _normalize_option_token(token: str) -> str:
    return token.strip("\"'`").strip("(){}[];,")


def _looks_like_script_executable(executable: str) -> bool:
    return executable.endswith(
        (
            ".bat",
            ".cmd",
            ".cjs",
            ".js",
            ".mjs",
            ".pl",
            ".ps1",
            ".py",
            ".rb",
            ".sh",
            ".zsh",
        )
    )


def _split_shell_segments(command: str) -> list[str]:
    return [
        segment.strip()
        for segment in command.replace("&&", "|")
        .replace("&", "|")
        .replace(";", "|")
        .replace("\r", "|")
        .replace("\n", "|")
        .split("|")
        if segment.strip()
    ]
