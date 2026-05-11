import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_CONFIG_FILE_BYTES = 2_097_152
MAX_CONFIG_TOOL_LIST_ITEMS = 128


def _ensure_existing_parent_directory(value: Path, field_name: str) -> Path:
    expanded = value.expanduser()
    for candidate in (expanded, *expanded.parents):
        try:
            exists = candidate.exists()
        except OSError as exc:
            raise ValueError(
                f"{field_name} path could not be checked: {value}"
            ) from exc
        if not exists:
            continue
        try:
            is_directory = candidate.is_dir()
        except OSError as exc:
            raise ValueError(
                f"{field_name} path could not be checked: {value}"
            ) from exc
        if not is_directory:
            raise ValueError(f"{field_name} must be a directory path: {value}")
        break
    return expanded


def _path_exists_or_error(path: Path, label: str) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        raise ValueError(f"{label} could not be checked: {path}") from exc


def _path_is_file_or_error(path: Path, label: str) -> bool:
    try:
        return path.is_file()
    except OSError as exc:
        raise ValueError(f"{label} could not be checked: {path}") from exc


def _ensure_file_size_or_error(path: Path, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"{label} could not be checked: {path}") from exc
    if size > MAX_CONFIG_FILE_BYTES:
        raise ValueError(f"{label} exceeds {MAX_CONFIG_FILE_BYTES} bytes: {path}")


def _reject_invalid_name_list(value: list[str], field_name: str) -> list[str]:
    if len(value) > MAX_CONFIG_TOOL_LIST_ITEMS:
        raise ValueError(
            f"{field_name} cannot contain more than "
            f"{MAX_CONFIG_TOOL_LIST_ITEMS} entries"
        )
    for name in value:
        if not name.strip():
            raise ValueError(f"{field_name} cannot contain empty names")
        if any(character.isspace() for character in name):
            raise ValueError(f"{field_name} cannot contain whitespace")
    return value


def _reject_invalid_module_list(value: list[str], field_name: str) -> list[str]:
    _reject_invalid_name_list(value, field_name)
    for name in value:
        if any(not part.isidentifier() for part in name.split(".")):
            raise ValueError(f"{field_name} must contain dotted Python module paths")
    return value


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    model: str = "gpt-5.5"
    base_url: str | None = None
    reasoning_effort: str = "low"
    text_verbosity: str = "low"
    max_steps: int = Field(default=20, gt=0)
    session_context_turns: int = Field(default=20, gt=0)
    session_summary_turns: int = Field(default=40, gt=0)
    workspace: Path = Path(".")
    permission_profile: Literal["standard", "read_only", "trusted"] = "standard"
    confirm_risky_actions: bool = True
    logs_dir: Path = Path("runs")
    shell_timeout_seconds: int = Field(default=30, gt=0)
    shell_max_output_bytes: int = Field(default=20000, gt=0, le=1_048_576)
    tool_modules: list[str] = Field(default_factory=list)
    enabled_tools: list[str] | None = None

    @field_validator("workspace")
    @classmethod
    def resolve_workspace(cls, value: Path) -> Path:
        expanded = _ensure_existing_parent_directory(value, "workspace")
        try:
            return expanded.resolve()
        except OSError as exc:
            raise ValueError(f"workspace path could not be resolved: {value}") from exc

    @field_validator("model", "reasoning_effort", "text_verbosity")
    @classmethod
    def reject_blank_model_settings(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} cannot be blank")
        return value

    @field_validator("logs_dir")
    @classmethod
    def reject_file_logs_dir(cls, value: Path) -> Path:
        _ensure_existing_parent_directory(value, "logs_dir")
        return value

    @field_validator("enabled_tools")
    @classmethod
    def reject_empty_tool_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return _reject_invalid_name_list(value, "enabled_tools")

    @field_validator("tool_modules")
    @classmethod
    def reject_empty_tool_modules(cls, value: list[str]) -> list[str]:
        return _reject_invalid_module_list(value, "tool_modules")


def dotenv_values(path: Path = Path(".env")) -> dict[str, str]:
    if not _path_exists_or_error(path, "Dotenv file"):
        return {}
    if not _path_is_file_or_error(path, "Dotenv file"):
        raise ValueError(f"Dotenv file must be a file: {path}")
    _ensure_file_size_or_error(path, "Dotenv file")

    values: dict[str, str] = {}
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Dotenv file could not be read as UTF-8: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Dotenv file could not be read: {path}") from exc
    if len(raw_text.encode("utf-8")) > MAX_CONFIG_FILE_BYTES:
        raise ValueError(f"Dotenv file exceeds {MAX_CONFIG_FILE_BYTES} bytes: {path}")
    lines = raw_text.splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _first_env(dotenv: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = os.environ.get(key) or dotenv.get(key)
        if value:
            return value
    return None


def load_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    dotenv = dotenv_values()
    data: dict[str, Any] = {}
    env_model = _first_env(dotenv, ("OPENAI_MODEL", "AZURE_OPENAI_DEPLOYMENT"))
    env_base_url = _first_env(
        dotenv,
        (
            "OPENAI_BASE_URL",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_EXISTING_AIPROJECT_ENDPOINT",
        ),
    )
    if env_model:
        data["model"] = env_model
    if env_base_url:
        data["base_url"] = env_base_url

    if config_path is not None:
        if not _path_exists_or_error(config_path, "Config file"):
            raise ValueError(f"Config file not found: {config_path}")
        if not _path_is_file_or_error(config_path, "Config file"):
            raise ValueError(f"Config file must be a file: {config_path}")
        _ensure_file_size_or_error(config_path, "Config file")
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Config file could not be read as UTF-8: {config_path}"
            ) from exc
        except OSError as exc:
            raise ValueError(f"Config file could not be read: {config_path}") from exc
        if len(config_text.encode("utf-8")) > MAX_CONFIG_FILE_BYTES:
            raise ValueError(
                f"Config file exceeds {MAX_CONFIG_FILE_BYTES} bytes: {config_path}"
            )
        try:
            loaded = yaml.safe_load(config_text) or {}
        except (yaml.YAMLError, RecursionError) as exc:
            raise ValueError(f"Config file could not be parsed: {config_path}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a mapping: {config_path}")
        data.update(loaded)

    if overrides:
        data.update(
            {key: value for key, value in overrides.items() if value is not None}
        )

    return RunConfig(**data)
