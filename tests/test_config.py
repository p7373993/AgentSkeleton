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
    assert config.enabled_tools is None


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
                "base_url: https://example.openai.azure.com/openai/v1/",
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
    assert config.base_url == "https://example.openai.azure.com/openai/v1/"
    assert config.enabled_tools == ["read_file", "ask_user"]
    assert config.workspace == workspace.resolve()


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


def test_config_rejects_non_positive_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, max_steps=0)

    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, shell_timeout_seconds=0)
