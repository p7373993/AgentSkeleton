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
            if not isinstance(command, str) or not command.strip():
                return PermissionDecision("block", "Shell command invalid")
            return self._decide_shell(command)

        if risk == "read" or risk == "interactive":
            return PermissionDecision("allow", "read-only or interactive tool")

        if self.profile == "read_only":
            return PermissionDecision(
                "block",
                "Tool is blocked by read-only profile",
            )

        return PermissionDecision("confirm", f"Unknown risk level: {risk}")

    def _decide_shell(self, command: str) -> PermissionDecision:
        normalized = command.lower()
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
        ]
        has_network_access = _contains_fetch_command(normalized) or any(
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


def _looks_like_recursive_delete(command: str) -> bool:
    delete_commands = ("remove-item", "rm", "ri", "del", "erase", "rd", "rmdir")
    tokens = command.replace(";", " ").replace("|", " ").split()
    if not any(token in delete_commands for token in tokens):
        return False
    return any(
        token in {"-recurse", "-r", "/s"} or _is_compact_recursive_flag(token)
        for token in tokens
    )


def _is_compact_recursive_flag(token: str) -> bool:
    return token.startswith("-") and len(token) <= 4 and "r" in token[1:]


def _looks_like_encoded_command(command: str) -> bool:
    tokens = command.replace(";", " ").replace("|", " ").split()
    encoded_flags = {"-encodedcommand", "-encoded", "-enco", "-enc", "-ec", "-e"}
    return any(token in encoded_flags for token in tokens)


def _looks_like_remote_execution(command: str) -> bool:
    executor_commands = (
        "bash",
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


def _contains_fetch_command(command: str) -> bool:
    return any(
        _segment_has_fetch_command(segment)
        for segment in _split_shell_segments(command)
    )


def _segment_has_fetch_command(segment: str) -> bool:
    return any(
        _normalize_executable_token(token) in FETCH_COMMANDS
        for token in segment.split()
    )


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
    executable = token.strip("\"'").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if executable.endswith(".exe"):
        executable = executable[:-4]
    return executable


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
