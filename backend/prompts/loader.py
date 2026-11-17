"""
Prompt loader — reads version-controlled prompt files from the prompts/ directory.

Files are cached per process. A missing or empty prompt file fails fast with
PromptNotFoundError so the app surfaces it at startup rather than silently
running with a blank system prompt.
"""
import hashlib
from pathlib import Path
from typing import Dict

_PROMPTS_DIR = Path(__file__).resolve().parent
_cache: Dict[str, str] = {}


class PromptNotFoundError(Exception):
    """A required prompt file is missing or empty."""


def load_prompt(name: str) -> str:
    """
    Read ``prompts/<name>`` as UTF-8 text.

    The result is cached for the lifetime of the process.  A missing or empty
    file raises :class:`PromptNotFoundError`.
    """
    if name in _cache:
        return _cache[name]

    path = _PROMPTS_DIR / name
    if not path.is_file():
        raise PromptNotFoundError(
            f"Prompt file not found: {path}  "
            f"(resolved from prompts/{name})"
        )

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PromptNotFoundError(
            f"Prompt file is empty: {path}"
        )

    _cache[name] = text
    return text


def prompt_version() -> str:
    """
    Return the first 12 hex chars of sha256 over the system + react-instruction
    prompt texts.  This gives a stable, short fingerprint that changes whenever
    either file is edited.
    """
    from config import get_config
    cfg = get_config()
    prompts_cfg = getattr(cfg, "prompts", None)
    sys_name = prompts_cfg.system if prompts_cfg else "system.md"
    react_name = prompts_cfg.react_instructions if prompts_cfg else "react_instructions.md"

    sys_text = load_prompt(sys_name)
    react_text = load_prompt(react_name)

    h = hashlib.sha256()
    h.update(sys_text.encode("utf-8"))
    h.update(react_text.encode("utf-8"))
    return h.hexdigest()[:12]


def _reset_cache() -> None:
    """Clear the prompt cache — for tests only."""
    _cache.clear()
