import os
from pathlib import Path

import pytest

import agentskeleton.config as config_module
from agentskeleton.config import RunConfig, load_config


class UnencodableString(str):
    def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("cannot encode")


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


def test_load_config_reports_config_exists_stat_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "agent.yaml"
    original_exists = Path.exists

    def fail_config_exists(path: Path) -> bool:
        if path == config_path:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_config_exists)

    with pytest.raises(ValueError, match="Config file could not be checked"):
        load_config(config_path)


def test_load_config_rejects_config_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.mkdir()

    with pytest.raises(ValueError, match="Config file must be a file"):
        load_config(config_path)


def test_load_config_reports_config_file_stat_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("model: gpt-5.4-mini\n", encoding="utf-8")
    original_is_file = Path.is_file

    def fail_config_is_file(path: Path) -> bool:
        if path == config_path:
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_config_is_file)

    with pytest.raises(ValueError, match="Config file could not be checked"):
        load_config(config_path)


def test_load_config_rejects_oversized_config_before_reading(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("x" * 11, encoding="utf-8")
    monkeypatch.setattr(config_module, "MAX_CONFIG_FILE_BYTES", 10)
    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs) -> str:
        if path == config_path:
            raise AssertionError("oversized config file should not be read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(ValueError, match=r"Config file exceeds 10 bytes"):
        load_config(config_path)


def test_load_config_rejects_config_that_grows_after_stat(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(config_module, "MAX_CONFIG_FILE_BYTES", 10)
    original_read_text = Path.read_text

    def grow_read_text(path: Path, *args, **kwargs) -> str:
        if path == config_path:
            return "x" * 11
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", grow_read_text)

    with pytest.raises(ValueError, match=r"Config file exceeds 10 bytes"):
        load_config(config_path)


def test_load_config_rejects_unencodable_config_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("model: gpt-5.4-mini\n", encoding="utf-8")
    original_read_text = Path.read_text

    def unencodable_read_text(path: Path, *args, **kwargs) -> str:
        if path == config_path:
            return UnencodableString("model: gpt-5.4-mini\n")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unencodable_read_text)

    with pytest.raises(ValueError, match=r"Config file could not be inspected"):
        load_config(config_path)


def test_load_config_reports_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("model: [unterminated\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Config file could not be parsed"):
        load_config(config_path)


def test_load_config_reports_deeply_nested_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("[" * 20_000 + "null" + "]" * 20_000, encoding="utf-8")

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


def test_load_config_reports_dotenv_exists_stat_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    original_exists = Path.exists

    def fail_dotenv_exists(path: Path) -> bool:
        if path == Path(".env"):
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_dotenv_exists)

    with pytest.raises(ValueError, match="Dotenv file could not be checked"):
        load_config()


def test_load_config_rejects_dotenv_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").mkdir()

    with pytest.raises(ValueError, match=r"Dotenv file must be a file"):
        load_config()


def test_load_config_reports_dotenv_file_stat_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_MODEL=gpt-5.4-mini\n", encoding="utf-8")
    original_is_file = Path.is_file

    def fail_dotenv_is_file(path: Path) -> bool:
        if path == Path(".env"):
            raise OSError("permission denied")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", fail_dotenv_is_file)

    with pytest.raises(ValueError, match="Dotenv file could not be checked"):
        load_config()


def test_load_config_rejects_oversized_dotenv_before_reading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("x" * 11, encoding="utf-8")
    monkeypatch.setattr(config_module, "MAX_CONFIG_FILE_BYTES", 10)
    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs) -> str:
        if path == Path(".env"):
            raise AssertionError("oversized dotenv file should not be read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(ValueError, match=r"Dotenv file exceeds 10 bytes"):
        load_config()


def test_load_config_rejects_dotenv_that_grows_after_stat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(config_module, "MAX_CONFIG_FILE_BYTES", 10)
    original_read_text = Path.read_text

    def grow_read_text(path: Path, *args, **kwargs) -> str:
        if path == Path(".env"):
            return "x" * 11
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", grow_read_text)

    with pytest.raises(ValueError, match=r"Dotenv file exceeds 10 bytes"):
        load_config()


def test_load_config_rejects_unencodable_dotenv_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_MODEL=gpt-5.4-mini\n", encoding="utf-8")
    original_read_text = Path.read_text

    def unencodable_read_text(path: Path, *args, **kwargs) -> str:
        if path == Path(".env"):
            return UnencodableString("OPENAI_MODEL=gpt-5.4-mini\n")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unencodable_read_text)

    with pytest.raises(ValueError, match=r"Dotenv file could not be inspected"):
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


def test_config_rejects_excessive_max_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, max_steps=201)


def test_config_rejects_oversized_shell_output_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, shell_max_output_bytes=1_048_577)


def test_config_rejects_unknown_permission_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig(workspace=tmp_path, permission_profile="reckless")


def test_config_rejects_blank_model_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model cannot be blank"):
        RunConfig(workspace=tmp_path, model="   ")

    with pytest.raises(ValueError, match="reasoning_effort cannot be blank"):
        RunConfig(workspace=tmp_path, reasoning_effort="   ")

    with pytest.raises(ValueError, match="text_verbosity cannot be blank"):
        RunConfig(workspace=tmp_path, text_verbosity="   ")


def test_config_rejects_whitespace_in_enabled_tools(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="enabled_tools cannot contain whitespace"):
        RunConfig(workspace=tmp_path, enabled_tools=[" read_file"])

    with pytest.raises(ValueError, match="enabled_tools cannot contain whitespace"):
        RunConfig(workspace=tmp_path, enabled_tools=["read file"])


def test_config_rejects_whitespace_in_tool_modules(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tool_modules cannot contain whitespace"):
        RunConfig(workspace=tmp_path, tool_modules=[" custom_tools"])

    with pytest.raises(ValueError, match="tool_modules cannot contain whitespace"):
        RunConfig(workspace=tmp_path, tool_modules=["custom tools"])


def test_config_rejects_too_many_enabled_tools(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="enabled_tools cannot contain more than 128 entries",
    ):
        RunConfig(
            workspace=tmp_path,
            enabled_tools=[f"tool_{index}" for index in range(129)],
        )


def test_config_rejects_too_many_tool_modules(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="tool_modules cannot contain more than 128 entries",
    ):
        RunConfig(
            workspace=tmp_path,
            tool_modules=[f"module_{index}" for index in range(129)],
        )


def test_config_rejects_oversized_enabled_tool_names(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="enabled_tools entries cannot exceed 512 bytes",
    ):
        RunConfig(workspace=tmp_path, enabled_tools=["a" * 513])


def test_config_name_list_rejects_unencodable_entries() -> None:
    with pytest.raises(
        ValueError,
        match="enabled_tools entries could not be inspected",
    ):
        config_module._reject_invalid_name_list(  # noqa: SLF001
            [UnencodableString("read_file")],
            "enabled_tools",
        )


def test_config_rejects_oversized_tool_module_names(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="tool_modules entries cannot exceed 512 bytes",
    ):
        RunConfig(workspace=tmp_path, tool_modules=["a" * 513])


def test_config_module_list_rejects_unencodable_entries() -> None:
    with pytest.raises(
        ValueError,
        match="tool_modules entries could not be inspected",
    ):
        config_module._reject_invalid_module_list(  # noqa: SLF001
            [UnencodableString("custom_tools")],
            "tool_modules",
        )


@pytest.mark.parametrize(
    "module_name",
    [
        "../tools.py",
        ".custom_tools",
        "custom_tools.",
        "custom..tools",
        "custom-tools",
    ],
)
def test_config_rejects_malformed_tool_modules(
    tmp_path: Path,
    module_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="tool_modules must contain dotted Python module paths",
    ):
        RunConfig(workspace=tmp_path, tool_modules=[module_name])


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


def test_config_reports_workspace_stat_failures(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    original_exists = Path.exists

    def fail_workspace_stat(path: Path) -> bool:
        if path == workspace:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_workspace_stat)

    with pytest.raises(ValueError, match="workspace path could not be checked"):
        RunConfig(workspace=workspace)


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


def test_config_reports_logs_dir_stat_failures(monkeypatch, tmp_path: Path) -> None:
    logs_dir = tmp_path / "runs"
    original_exists = Path.exists

    def fail_logs_dir_stat(path: Path) -> bool:
        if path == logs_dir:
            raise OSError("permission denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_logs_dir_stat)

    with pytest.raises(ValueError, match="logs_dir path could not be checked"):
        RunConfig(workspace=tmp_path, logs_dir=logs_dir)
