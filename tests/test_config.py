import os
from pathlib import Path

import pytest

from agentskeleton.config import RunConfig, load_config


def test_default_config_uses_current_directory(tmp_path: Path) -> None:
    config = RunConfig(workspace=tmp_path)

    assert config.model == "gpt-5.5"
    assert config.reasoning_effort == "low"
    assert config.text_verbosity == "low"
    assert config.max_steps == 20
    assert config.workspace == tmp_path.resolve()
    assert config.logs_dir == Path("runs")
    assert config.shell_timeout_seconds == 30
    assert config.shell_max_output_bytes == 20000
    assert config.session_context_turns == 20
    assert config.session_summary_turns == 40
    assert config.tool_modules == []
    assert config.enabled_tools is None
    assert config.permission_profile == "standard"


def test_load_config_merges_yaml_and_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path.write_text(
        "\n".join(
            [
                "model: gpt-5.4-mini",
                "reasoning_effort: medium",
                "max_steps: 7",
                "session_context_turns: 5",
                "session_summary_turns: 9",
                "base_url: https://example.openai.azure.com/openai/v1/",
                "permission_profile: read_only",
                "tool_modules:",
                "  - custom_tools",
                "enabled_tools:",
                "  - read_file",
                "  - ask_user",
                f"workspace: {workspace.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path, {"max_steps": 3})

    assert config.model == "gpt-5.4-mini"
    assert config.reasoning_effort == "medium"
    assert config.max_steps == 3
    assert config.session_context_turns == 5
    assert config.session_summary_turns == 9
    assert config.base_url == "https://example.openai.azure.com/openai/v1/"
    assert config.permission_profile == "read_only"
    assert config.tool_modules == ["custom_tools"]
    assert config.enabled_tools == ["read_file", "ask_user"]
    assert config.workspace == workspace.resolve()


def test_load_config_rejects_explicit_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Config file not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_config_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.mkdir()

    with pytest.raises(ValueError, match="Config file must be a file"):
        load_config(config_path)


def test_load_config_reports_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("model: [unterminated\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Config file could not be parsed"):
        load_config(config_path)


def test_load_config_reports_invalid_utf8_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_bytes(b"\xff\xfe\x00broken")

    with pytest.raises(ValueError, match=r"Config file could not be read as UTF-8"):
        load_config(config_path)


def test_load_config_uses_model_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")

    config = load_config()

    assert config.model == "gpt-5.4-mini"


def test_load_config_reads_dotenv_without_overriding_real_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_EXISTING_AIPROJECT_ENDPOINT", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "from-real-env")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                'AZURE_OPENAI_DEPLOYMENT="from-dotenv"',
                'AZURE_EXISTING_AIPROJECT_ENDPOINT="https://example/openai/v1"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config()

    assert config.model == "from-real-env"
    assert config.base_url == "https://example/openai/v1"
    assert "AZURE_EXISTING_AIPROJECT_ENDPOINT" not in os.environ


def test_load_config_reports_invalid_utf8_dotenv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_bytes(b"\xff\xfe\x00broken")

    with pytest.raises(ValueError, match=r"Dotenv file could not be read as UTF-8"):
        load_config()


def test_load_config_rejects_dotenv_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").mkdir()

    with pytest.raises(ValueError, match=r"Dotenv file must be a file"):
        load_config()


def test_config_rejects_non_positive_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, max_steps=0)

    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, shell_timeout_seconds=0)

    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, session_context_turns=0)

    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, session_summary_turns=0)


def test_config_rejects_unknown_permission_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, permission_profile="reckless")


def test_config_rejects_workspace_file(tmp_path: Path) -> None:
    workspace_file = tmp_path / "workspace"
    workspace_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="workspace must be a directory path"):
        RunConfig(workspace=workspace_file)


def test_config_rejects_workspace_with_file_parent(tmp_path: Path) -> None:
    parent_file = tmp_path / "workspace"
    parent_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="workspace must be a directory path"):
        RunConfig(workspace=parent_file / "child")


def test_config_rejects_logs_dir_file(tmp_path: Path) -> None:
    logs_file = tmp_path / "runs"
    logs_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="logs_dir must be a directory"):
        RunConfig(workspace=tmp_path, logs_dir=logs_file)


def test_config_rejects_logs_dir_with_file_parent(tmp_path: Path) -> None:
    parent_file = tmp_path / "runs"
    parent_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="logs_dir must be a directory path"):
        RunConfig(workspace=tmp_path, logs_dir=parent_file / "nested")
