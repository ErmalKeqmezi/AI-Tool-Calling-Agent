"""Application settings, loaded from environment variables (and an optional .env file).

Secrets are only ever read from the environment - never hardcoded, never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from the project root if it exists. Real environment variables win.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    anthropic_api_key: str | None = field(repr=False, default=None)  # repr=False: never printed
    model: str = "claude-opus-5"
    max_tokens: int = 16000
    request_timeout: float = 60.0  # seconds per HTTP request to the LLM
    max_retries: int = 2  # SDK-level retries for 429 / 5xx / network errors
    refusal_fallback: bool = True  # server-side fallback model if the main model refuses

    # --- Agent ---
    max_agent_iterations: int = 10  # hard cap on LLM round-trips per user message
    max_history_turns: int = 20  # short-term memory: user turns kept in context

    # --- Tools ---
    tool_http_timeout: float = 10.0
    tavily_api_key: str | None = field(repr=False, default=None)
    file_sandbox_dir: Path = PROJECT_ROOT / "workspace"

    # --- Logging ---
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            model=os.getenv("LLM_MODEL") or cls.model,
            max_tokens=_env_int("LLM_MAX_TOKENS", cls.max_tokens),
            request_timeout=_env_float("LLM_TIMEOUT_SECONDS", cls.request_timeout),
            max_retries=_env_int("LLM_MAX_RETRIES", cls.max_retries),
            refusal_fallback=_env_bool("LLM_REFUSAL_FALLBACK", cls.refusal_fallback),
            max_agent_iterations=_env_int("AGENT_MAX_ITERATIONS", cls.max_agent_iterations),
            max_history_turns=_env_int("AGENT_MAX_HISTORY_TURNS", cls.max_history_turns),
            tool_http_timeout=_env_float("TOOL_HTTP_TIMEOUT_SECONDS", cls.tool_http_timeout),
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            file_sandbox_dir=Path(os.getenv("FILE_SANDBOX_DIR") or PROJECT_ROOT / "workspace"),
            log_level=(os.getenv("LOG_LEVEL") or cls.log_level).upper(),
        )

    def require_api_key(self) -> str:
        if not self.anthropic_api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export ANTHROPIC_API_KEY in your shell."
            )
        return self.anthropic_api_key


def get_settings() -> Settings:
    return Settings.from_env()
