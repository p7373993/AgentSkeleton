from dataclasses import dataclass
from typing import Any, Literal

PermissionOutcome = Literal["allow", "confirm", "block"]
PermissionProfile = Literal["standard", "read_only", "trusted"]


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
        if risk == "read" or risk == "interactive":
            return PermissionDecision("allow", "read-only or interactive tool")

        if self.profile == "read_only":
            return PermissionDecision(
                "block",
                "Tool is blocked by read-only profile",
            )

        if tool_name == "write_file" or risk == "write":
            if self.confirm_risky_actions and self.profile != "trusted":
                return PermissionDecision(
                    "confirm",
                    "Tool writes files in the workspace",
                )
            return PermissionDecision("allow", "file writes allowed by configuration")

        if tool_name == "shell" or risk == "shell":
            return self._decide_shell(str(args.get("command", "")))

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
            "password",
            "secret",
        ]
        network_patterns = [
            "curl ",
            "curl.exe",
            "wget ",
            "invoke-webrequest",
            "invoke-restmethod",
            "iwr ",
            "irm ",
            "requests.",
            "urllib.",
            "http.client",
        ]
        if any(net in normalized for net in network_patterns) and any(
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
            "npm install",
            "pnpm install",
            "yarn add",
            "curl ",
            "wget ",
            "git push",
            "start ",
        ]
        if (
            self.confirm_risky_actions
            and self.profile != "trusted"
            and any(pattern in normalized for pattern in confirm_patterns)
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
    return any(token in {"-encodedcommand", "-enc"} for token in tokens)


def _looks_like_remote_execution(command: str) -> bool:
    fetch_patterns = (
        "curl ",
        "curl.exe",
        "wget ",
        "invoke-webrequest",
        "invoke-restmethod",
        "iwr ",
        "irm ",
    )
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
    segments = [segment.strip() for segment in command.split("|")]
    for index, segment in enumerate(segments[:-1]):
        if not any(pattern in segment for pattern in fetch_patterns):
            continue
        next_tokens = segments[index + 1].split()
        if next_tokens and next_tokens[0] in executor_commands:
            return True
    return False
