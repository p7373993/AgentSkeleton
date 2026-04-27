from agentskeleton.policy.permissions import PermissionPolicy


def test_policy_allows_read_only_tools() -> None:
    decision = PermissionPolicy().decide("read_file", {"path": "README.md"}, "read")

    assert decision.outcome == "allow"


def test_policy_confirms_write_file_by_default() -> None:
    decision = PermissionPolicy(confirm_risky_actions=True).decide(
        "write_file",
        {"path": "README.md", "content": "hello"},
        "write",
    )

    assert decision.outcome == "confirm"
    assert "writes files" in decision.reason


def test_policy_allows_write_when_confirmation_disabled() -> None:
    decision = PermissionPolicy(confirm_risky_actions=False).decide(
        "write_file",
        {"path": "README.md", "content": "hello"},
        "write",
    )

    assert decision.outcome == "allow"


def test_policy_blocks_clearly_destructive_shell_commands() -> None:
    decision = PermissionPolicy().decide("shell", {"command": "rm -rf /"}, "shell")

    assert decision.outcome == "block"
    assert "destructive" in decision.reason


def test_policy_confirms_package_install_shell_commands() -> None:
    decision = PermissionPolicy().decide(
        "shell",
        {"command": "uv pip install pytest"},
        "shell",
    )

    assert decision.outcome == "confirm"
    assert "install" in decision.reason
