"""
Centralised configuration using Pydantic BaseSettings.

Values are read from environment variables or the .env file at project root.
All other modules import `settings` from here — no ad-hoc os.getenv() calls.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Valid OpenAI model name aliases: allow the user to type them naturally
# (e.g. "GPT-4.1 mini", "gpt4o", "gpt-4o mini") and normalise to the
# canonical kebab-case IDs the API actually accepts.
_MODEL_ALIASES: dict[str, str] = {
    # GPT-4o family
    "gpt4o":        "gpt-4o",
    "gpt-4o":       "gpt-4o",
    "gpt-4o-mini":  "gpt-4o-mini",
    "gpt4omini":    "gpt-4o-mini",
    # GPT-4.1 family
    "gpt-4.1":      "gpt-4.1",
    "gpt4.1":       "gpt-4.1",
    "gpt-4.1mini":  "gpt-4.1-mini",
    "gpt4.1mini":   "gpt-4.1-mini",
    "gpt-4.1-mini": "gpt-4.1-mini",
    # GPT-5 family (reasoning models — 400k context window)
    "gpt-5":        "gpt-5",
    "gpt5":         "gpt-5",
    "gpt-5-mini":   "gpt-5-mini",
    "gpt5mini":     "gpt-5-mini",
    "gpt5-mini":    "gpt-5-mini",
}

# Context window sizes per model family — used to auto-configure the guard
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o":       128_000,
    "gpt-4o-mini":  128_000,
    "gpt-4.1":      128_000,
    "gpt-4.1-mini": 128_000,
    "gpt-5":        400_000,
    "gpt-5-mini":   400_000,
}


class Settings(BaseSettings):
    # ── OpenAI / LLM ─────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4o", env="MODEL_NAME")

    @field_validator("model_name", mode="before")
    @classmethod
    def normalise_model_name(cls, v: str) -> str:
        """
        Normalise the model name read from .env so common typos are accepted.

        Transformations applied (in order):
          1. strip surrounding whitespace
          2. lowercase
          3. collapse consecutive spaces/hyphens to a single hyphen
          4. look up known aliases

        Examples
        --------
        "GPT-4.1 mini"  →  "gpt-4.1-mini"
        "gpt4o"         →  "gpt-4o"
        "GPT-4o"        →  "gpt-4o"
        """
        import re
        normalised = re.sub(r"[\s\-]+", "-", v.strip().lower())
        # Look up alias table; fall back to the normalised string itself
        return _MODEL_ALIASES.get(normalised, normalised)
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")

    # ── Pipeline control ──────────────────────────────────────────────────────
    max_iterations: int = Field(default=5, env="MAX_ITERATIONS")
    convergence_threshold: float = Field(default=0.5, env="CONVERGENCE_THRESHOLD")

    # ── Context window guard ──────────────────────────────────────────────────
    # Default 0 means "auto-detect from model name"; set explicitly in .env to override.
    context_window_limit: int = Field(default=0, env="CONTEXT_WINDOW_LIMIT")

    @field_validator("context_window_limit", mode="after")
    @classmethod
    def auto_context_window(cls, v: int, info) -> int:
        """If context_window_limit is 0 (default), infer it from the model name."""
        if v != 0:
            return v
        model = info.data.get("model_name", "gpt-4o")
        return _MODEL_CONTEXT_WINDOWS.get(model, 128_000)
    context_window_guard_pct: float = Field(default=0.8, env="CONTEXT_WINDOW_GUARD_PCT")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path: str = Field(default="data/traffic_data.csv", env="DATA_PATH")

    # ── LangSmith Observability (all optional — tracing is disabled if absent) ─
    langchain_tracing_v2: str  = Field(default="false",                   env="LANGCHAIN_TRACING_V2")
    langchain_api_key:    str  = Field(default="",                        env="LANGCHAIN_API_KEY")
    langchain_project:    str  = Field(default="network-traffic-prediction", env="LANGCHAIN_PROJECT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Module-level singleton — import this everywhere
settings: Settings = get_settings()
