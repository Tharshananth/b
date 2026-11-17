"""
Process-wide, thread-safe model health and cooldown tracker.

Tracks LLM model availability and cooldowns triggered by quota exhaustion,
model not found, or temporary service unavailability.
"""
import logging
import re
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_known_models: Set[str] = set()
_cooldowns: Dict[str, float] = {}  # model -> monotonic expiry time
_cooldown_reasons: Dict[str, str] = {}  # model -> reason
_warned_cooldowns: Dict[str, float] = {}  # model -> monotonic expiry for which warning was logged


def register_model(model: str) -> None:
    """Register a known model for health tracking."""
    if not model:
        return
    with _lock:
        _known_models.add(model)


def _sanitize_reason(reason: str) -> str:
    """Strip any accidental API keys from reason strings."""
    if not reason:
        return ""
    # Mask common API key patterns (Google AIza..., OpenAI sk-...)
    sanitized = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", reason)
    sanitized = re.sub(r"sk-[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", sanitized)
    return sanitized


def is_available(model: str) -> bool:
    """Check if a model is currently available (not in cooldown)."""
    register_model(model)
    with _lock:
        expiry = _cooldowns.get(model, 0.0)
        return time.monotonic() >= expiry


def mark_unavailable(model: str, seconds: float, reason: str) -> None:
    """Mark a model as unavailable for `seconds`, logging a warning once per cooldown."""
    register_model(model)
    with _lock:
        now = time.monotonic()
        until = now + max(0.0, float(seconds))
        _cooldowns[model] = until
        _cooldown_reasons[model] = reason

        # Log at WARNING ONCE per cooldown period
        last_warned_until = _warned_cooldowns.get(model, 0.0)
        if until > last_warned_until or now >= last_warned_until:
            _warned_cooldowns[model] = until
            safe_reason = _sanitize_reason(reason)
            logger.warning(
                "Model '%s' marked unavailable for %ds. Reason: %s",
                model,
                int(seconds),
                safe_reason,
            )


def mark_ok(model: str) -> None:
    """Mark a model as available immediately (clears cooldown)."""
    register_model(model)
    with _lock:
        _cooldowns.pop(model, None)
        _cooldown_reasons.pop(model, None)
        _warned_cooldowns.pop(model, None)


def get_cooldown_for_exception(exc: Exception) -> Optional[Tuple[int, str]]:
    """Determine cooldown duration and reason from an exception.

    Rules:
      - 429 / RESOURCE_EXHAUSTED: retryDelay if present, else 60s
      - 404 / NOT_FOUND: 3600s
      - 503 / UNAVAILABLE / 500 / timeout: 30s
      - 400 / 401 / 403 and others: None (not a model availability issue)
    """
    code = getattr(exc, "code", getattr(exc, "status_code", None))
    msg = str(exc)
    status_str = getattr(exc, "status", "")

    # Non-cooldown request/key bugs: 400 / 401 / 403
    if code in (400, 401, 403):
        return None
    for err_code in ("400", "401", "403"):
        if re.search(rf"\b{err_code}\b", msg) and not any(
            c in msg for c in ("429", "404", "500", "502", "503", "504")
        ):
            return None
    if any(
        k in msg for k in ("INVALID_ARGUMENT", "PERMISSION_DENIED", "UNAUTHENTICATED")
    ):
        return None

    # 429 / RESOURCE_EXHAUSTED
    if (
        code == 429
        or "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or status_str == "RESOURCE_EXHAUSTED"
    ):
        # Look for retryDelay, e.g. "retryDelay": "120s" or retryDelay 120s or 67789s
        match = re.search(r"retry[Dd]elay[\"']?\s*[:= ]\s*[\"']?(\d+)", msg)
        if match:
            seconds = int(match.group(1))
        else:
            seconds = 60
        return (seconds, "429 RESOURCE_EXHAUSTED")

    # 404 / NOT_FOUND
    if code == 404 or "404" in msg or "NOT_FOUND" in msg or status_str == "NOT_FOUND":
        return (3600, "404 NOT_FOUND")

    # 503 / UNAVAILABLE / 500 / timeout
    if (
        code in (500, 502, 503, 504)
        or any(c in msg for c in ("500", "502", "503", "504", "UNAVAILABLE"))
        or "timeout" in msg.lower()
        or "timed out" in msg.lower()
        or isinstance(exc, (TimeoutError,))
    ):
        return (30, "503/500/timeout")

    return None


def _ensure_config_models():
    """Lazily seed known models from config."""
    try:
        from config import get_config
        cfg = get_config()
        if "gemini" in cfg.llm.providers:
            gp = cfg.llm.providers["gemini"]
            register_model(gp.model)
            for m in getattr(gp, "fallback_models", []):
                register_model(m)
        if hasattr(cfg, "guardrails") and hasattr(cfg.guardrails, "medical"):
            gr = cfg.guardrails.medical
            register_model(gr.model)
            for m in getattr(gr, "fallback_models", []):
                register_model(m)
    except Exception:
        pass


def status() -> Dict[str, Dict[str, Any]]:
    """Return health status dictionary of all known models."""
    _ensure_config_models()
    with _lock:
        now = time.monotonic()
        models_to_report = set(_known_models)
        models_to_report.update(_cooldowns.keys())

        result: Dict[str, Dict[str, Any]] = {}
        for model in sorted(models_to_report):
            expiry = _cooldowns.get(model, 0.0)
            if now < expiry:
                rem = int(expiry - now)
                result[model] = {
                    "status": "cooling_down",
                    "seconds_remaining": rem,
                    "reason": _cooldown_reasons.get(model, "cooling down"),
                }
            else:
                result[model] = {
                    "status": "available",
                    "seconds_remaining": 0,
                }
        return result


def reset_health() -> None:
    """Clear all health states (for testing)."""
    with _lock:
        _cooldowns.clear()
        _cooldown_reasons.clear()
        _warned_cooldowns.clear()
        _known_models.clear()
