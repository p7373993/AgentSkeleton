import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "gpt-5.5"
    base_url: str | None = None
    reasoning_effort: str = "low"
    text_verbosity: str = "low"
    max_steps: int = Field(default=20, gt=0)
    workspace: Path = Path(".")
    confirm_risky_actions: bool = True
    logs_dir: Path = Path("runs")
    shell_timeout_seconds: int = Field(default=30, gt=0)
    shell_max_output_bytes: int = Field(default=20000, gt=0)

    @field_validator("workspace")
    @classmethod
    def resolve_workspace(cls, value: Path) -> Path:
        return value.expanduser().resolve()


def dotenv_values(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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

    if config_path is not None and config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a mapping: {config_path}")
        data.update(loaded)

    if overrides:
        data.update(
            {key: value for key, value in overrides.items() if value is not None}
        )

    return RunConfig(**data)
