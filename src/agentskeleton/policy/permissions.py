from dataclasses import dataclass
from typing import Any, Literal

PermissionOutcome = Literal["allow", "confirm", "block"]


@dataclass(frozen=True)
class PermissionDecision:
    outcome: PermissionOutcome
    reason: str


class PermissionPolicy:
    def __init__(self, confirm_risky_actions: bool = True) -> None:
        self.confirm_risky_actions = confirm_risky_actions

    def decide(
        self,
        tool_name: str,
        args: dict[str, Any],
        risk: str,
    ) -> PermissionDecision:
        if risk == "read" or risk == "interactive":
            return PermissionDecision("allow", "read-only or interactive tool")

        if tool_name == "write_file" or risk == "write":
            if self.confirm_risky_actions:
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

        secret_patterns = [
            ".env",
            "id_rsa",
            "api_key",
            "apikey",
            "password",
            "secret",
        ]
        network_patterns = ["curl ", "wget ", "invoke-webrequest", "iwr "]
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
        if self.confirm_risky_actions and any(
            pattern in normalized for pattern in confirm_patterns
        ):
            return PermissionDecision(
                "confirm",
                "Shell command may install, network, or spawn",
            )

        return PermissionDecision("allow", "shell command allowed")
