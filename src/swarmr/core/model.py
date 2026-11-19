"""Shared model wiring.

One place decides which LLM every team uses. Teams never construct a model
themselves, so swapping provider is configuration, not code: `SWARMR_MODEL`
names any provider LangChain knows as `<provider>:<model>`, and the provider's
own key variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) authenticates it.
Without `SWARMR_MODEL` the model is Kimi's coding plan, configured by `KIMI_*`.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "build_model", "load_env"]

# Kimi's coding-plan endpoint. Deliberately not api.moonshot.ai: a
# kimi-for-coding key is rejected there with HTTP 401.
DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "kimi-for-coding"

# What a .env may contribute: this package's own settings, Kimi's, and any
# provider's key. Nothing else in the file reaches the process.
_PREFIXES = ("SWARMR_", "KIMI_")
_KEY_SUFFIX = "_API_KEY"

_TIMEOUT = 180
_RETRIES = 3


def _configured() -> bool:
    """Whether the environment already says which model to use."""
    return bool(os.environ.get("SWARMR_MODEL") or os.environ.get("KIMI_API_KEY"))


def load_env() -> None:
    """Load the model settings from the nearest .env, walking up from cwd, then $HOME.

    Read: `SWARMR_*`, `KIMI_*` and every `*_API_KEY`. No dotenv dependency, and
    no path baked in. `SWARMR_ENV_FILE` (or `KIMI_ENV_FILE`) overrides. An
    environment that already names a model is authoritative: no file is read.
    """
    if _configured():
        return
    candidates: list[Path] = []
    for name in ("SWARMR_ENV_FILE", "KIMI_ENV_FILE"):
        if override := os.environ.get(name):
            candidates.append(Path(override))
    candidates += [p / ".env" for p in (Path.cwd(), *Path.cwd().parents)]
    candidates.append(Path.home() / ".env")

    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith(_PREFIXES) or key.endswith(_KEY_SUFFIX):
                os.environ.setdefault(key, value.strip().strip("\"'"))
        if _configured():
            return


def build_model() -> BaseChatModel:
    """The shared chat model.

    `SWARMR_MODEL=<provider>:<model>` selects any provider whose LangChain
    integration is installed; the provider reads its own key from the
    environment and names the variable when it is missing. Otherwise Kimi.

    Temperature is left unset on purpose: kimi-for-coding rejects any value
    but 1, so passing 0 fails the request with HTTP 400, and every other
    provider's default is the one its models are tuned for.
    """
    load_env()
    if spec := os.environ.get("SWARMR_MODEL", "").strip():
        if ":" not in spec:
            raise RuntimeError(
                f"SWARMR_MODEL={spec!r} must be <provider>:<model>, e.g. "
                "anthropic:<model id> with ANTHROPIC_API_KEY set."
            )
        return init_chat_model(spec, timeout=_TIMEOUT, max_retries=_RETRIES)
    key = os.environ.get("KIMI_API_KEY")
    if not key:
        raise RuntimeError(
            "No model is configured and no .env naming one was found. Set "
            "SWARMR_MODEL=<provider>:<model> with that provider's API key, or "
            "KIMI_API_KEY; SWARMR_ENV_FILE points at the file holding them."
        )
    return ChatOpenAI(
        model=os.environ.get("KIMI_MODEL", DEFAULT_MODEL),
        api_key=SecretStr(key),
        base_url=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL),
        timeout=_TIMEOUT,
        max_retries=_RETRIES,
    )
