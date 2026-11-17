"""
Medical-advice guardrail for the BSAI chatbot.

Classifies every incoming user message BEFORE the orchestrator runs.
If the message asks for personal medical advice or clinical interpretation,
the guardrail blocks the request and returns a fixed canned reply.
No user question or answer text is ever logged.
"""
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from llm.base import Message

logger = logging.getLogger(__name__)

BLOCK_CATEGORIES = {"personal_medical", "clinical_interpretation"}

# Paths relative to this file so they work from any CWD
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_CLASSIFIER_PROMPT_PATH = _PROMPTS_DIR / "guardrail_classifier.md"
_OUTPUT_CHECK_PROMPT_PATH = _PROMPTS_DIR / "guardrail_output_check.md"


# ── Enums & schemas ──────────────────────────────────────────────────────────

class Category(str, Enum):
    technical = "technical"
    data_lookup = "data_lookup"
    greeting = "greeting"
    out_of_scope = "out_of_scope"
    personal_medical = "personal_medical"
    clinical_interpretation = "clinical_interpretation"


class Verdict(BaseModel):
    category: Category
    reason: str


class OutputVerdict(BaseModel):
    gives_medical_advice: bool
    reason: str


# ── Decision dataclass ───────────────────────────────────────────────────────

@dataclass
class GuardrailDecision:
    blocked: bool
    category: Optional[str]
    reason: str
    latency_ms: float
    error: Optional[str]
    stage: str  # "input" | "output"
    model: Optional[str] = None


# ── Main guardrail class ─────────────────────────────────────────────────────

class MedicalGuardrail:
    """
    Thread-safe medical-advice guardrail backed by gemini-3.5-flash-lite.

    The genai.Client is created lazily once per process, lock-guarded.
    Prompt templates are loaded once at construction time.
    No user text is ever written to log files.
    """

    def __init__(self, cfg=None):
        from config import get_guardrails_config
        self._cfg = cfg or get_guardrails_config().medical

        # Load prompt files once
        self._classifier_prompt: str = _CLASSIFIER_PROMPT_PATH.read_text(encoding="utf-8")
        self._output_check_prompt: str = _OUTPUT_CHECK_PROMPT_PATH.read_text(encoding="utf-8")

        self._client = None
        self._client_lock = threading.Lock()
        self._key_logged_missing = False

    # ── Client initialisation ────────────────────────────────────────────────

    def _get_client(self):
        """Lazy, thread-safe genai.Client creation."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            key = os.environ.get("GOOGLE_API_KEY")
            if not key:
                if not self._key_logged_missing:
                    logger.error(
                        "GOOGLE_API_KEY is not set; medical guardrail will use on_error policy"
                    )
                    self._key_logged_missing = True
                return None
            from google import genai
            self._client = genai.Client(api_key=key)
            return self._client

    # ── Contents builder (spec §3 / pretest build_contents pattern) ──────────

    def _build_contents(self, message: str, history: "List[Message]"):
        """
        Build the Gemini `contents` list from recent conversation history
        plus the current message.

        Only user/assistant TEXT turns are included (tool messages are skipped).
        At most `history_turns` user/assistant pairs are taken.
        The list always ends on a "user" turn.
        """
        from google.genai import types

        k = self._cfg.history_turns
        # Collect eligible (role, text) pairs from history
        pairs: List[tuple] = []
        for msg in history:
            if msg.role not in ("user", "assistant"):
                continue
            # Skip assistant messages that only carry tool_calls (no text)
            if msg.role == "assistant" and not msg.content:
                continue
            pairs.append((msg.role, msg.content))

        # Keep the last k user/assistant PAIRS (2 messages per pair)
        # Walk backwards to find up to k user turns with their preceding
        # assistant reply; simplest correct approach: take last 2*k items.
        pairs = pairs[-(k * 2):]

        contents = []
        for role, text in pairs:
            genai_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(role=genai_role, parts=[types.Part(text=text)])
            )

        # Remove any trailing model turns so the final turn is always user
        while contents and contents[-1].role == "model":
            contents.pop()

        # Append current message as the last user turn
        contents.append(
            types.Content(role="user", parts=[types.Part(text=message)])
        )
        return contents

    # ── Input check ─────────────────────────────────────────────────────────

    def check_input(self, message: str, history: "List[Message]") -> GuardrailDecision:
        """
        Classify the user message.  Never raises; on any error returns an
        error decision respecting on_error policy.
        """
        t0 = time.perf_counter()
        try:
            return self._do_check_input(message, history, t0)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error("Guardrail input check failed: %s", exc)
            blocked = self._cfg.on_error == "block"
            return GuardrailDecision(
                blocked=blocked,
                category=None,
                reason="classifier error",
                latency_ms=latency_ms,
                error=str(exc),
                stage="input",
            )

    def _do_check_input(
        self, message: str, history: "List[Message]", t0: float
    ) -> GuardrailDecision:
        from google.genai import types

        client = self._get_client()
        if client is None:
            latency_ms = (time.perf_counter() - t0) * 1000
            blocked = self._cfg.on_error == "block"
            return GuardrailDecision(
                blocked=blocked,
                category=None,
                reason="GOOGLE_API_KEY not set",
                latency_ms=latency_ms,
                error="GOOGLE_API_KEY not set",
                stage="input",
            )

        timeout_ms = int(self._cfg.timeout_s * 1000)
        cfg = types.GenerateContentConfig(
            system_instruction=self._classifier_prompt,
            response_mime_type="application/json",
            response_schema=Verdict,
            temperature=0.0,
            max_output_tokens=200,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

        contents = self._build_contents(message, history)

        from llm.model_health import (
            get_cooldown_for_exception,
            is_available,
            mark_ok,
            mark_unavailable,
            register_model,
        )

        all_models = [self._cfg.model] + list(getattr(self._cfg, "fallback_models", []))
        for m in all_models:
            register_model(m)

        candidates = [m for m in all_models if is_available(m)]
        if not candidates:
            candidates = list(all_models)

        last_exc = None
        for candidate_model in candidates:
            try:
                resp = client.models.generate_content(
                    model=candidate_model,
                    contents=contents,
                    config=cfg,
                )

                latency_ms = (time.perf_counter() - t0) * 1000

                # Parse response
                parsed = getattr(resp, "parsed", None)
                if parsed is not None:
                    cat_val = getattr(parsed.category, "value", parsed.category)
                    cat_str = cat_val.value if hasattr(cat_val, "value") else str(cat_val)
                    reason = getattr(parsed, "reason", "")
                else:
                    data = json.loads(resp.text)
                    cat_str = str(data.get("category", ""))
                    reason = str(data.get("reason", ""))

                # Validate category
                valid_cats = {c.value for c in Category}
                if cat_str not in valid_cats:
                    blocked = self._cfg.on_error == "block"
                    return GuardrailDecision(
                        blocked=blocked,
                        category=None,
                        reason=f"unknown category: {cat_str}",
                        latency_ms=latency_ms,
                        error=f"unknown category returned: {cat_str}",
                        stage="input",
                        model=candidate_model,
                    )

                mark_ok(candidate_model)
                blocked = cat_str in BLOCK_CATEGORIES
                return GuardrailDecision(
                    blocked=blocked,
                    category=cat_str,
                    reason=reason,
                    latency_ms=latency_ms,
                    error=None,
                    stage="input",
                    model=candidate_model,
                )
            except Exception as exc:
                last_exc = exc
                cooldown = get_cooldown_for_exception(exc)
                if cooldown:
                    mark_unavailable(candidate_model, cooldown[0], cooldown[1])
                    logger.warning(
                        "Gatekeeper model '%s' hit cooldown (%s); trying fallback",
                        candidate_model,
                        cooldown[1],
                    )
                    continue
                else:
                    logger.error("Gatekeeper model '%s' non-cooldown error: %s", candidate_model, exc)
                    raise

        if last_exc is not None:
            raise last_exc

    # ── Output check ─────────────────────────────────────────────────────────

    def check_output(self, answer: str) -> GuardrailDecision:
        """
        Check whether the assistant's answer contains medical advice.
        On error: blocked=False (input check already ran), error is set.
        Never raises.
        """
        t0 = time.perf_counter()
        try:
            return self._do_check_output(answer, t0)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error("Guardrail output check failed: %s", exc)
            return GuardrailDecision(
                blocked=False,
                category=None,
                reason="output check error",
                latency_ms=latency_ms,
                error=str(exc),
                stage="output",
            )

    def _do_check_output(self, answer: str, t0: float) -> GuardrailDecision:
        from google.genai import types

        client = self._get_client()
        if client is None:
            latency_ms = (time.perf_counter() - t0) * 1000
            return GuardrailDecision(
                blocked=False,
                category=None,
                reason="GOOGLE_API_KEY not set",
                latency_ms=latency_ms,
                error="GOOGLE_API_KEY not set",
                stage="output",
            )

        timeout_ms = int(self._cfg.timeout_s * 1000)
        cfg = types.GenerateContentConfig(
            system_instruction=self._output_check_prompt,
            response_mime_type="application/json",
            response_schema=OutputVerdict,
            temperature=0.0,
            max_output_tokens=200,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

        contents = [
            types.Content(role="user", parts=[types.Part(text=answer)])
        ]

        from llm.model_health import (
            get_cooldown_for_exception,
            is_available,
            mark_ok,
            mark_unavailable,
            register_model,
        )

        all_models = [self._cfg.model] + list(getattr(self._cfg, "fallback_models", []))
        for m in all_models:
            register_model(m)

        candidates = [m for m in all_models if is_available(m)]
        if not candidates:
            candidates = list(all_models)

        last_exc = None
        for candidate_model in candidates:
            try:
                resp = client.models.generate_content(
                    model=candidate_model,
                    contents=contents,
                    config=cfg,
                )

                latency_ms = (time.perf_counter() - t0) * 1000

                parsed = getattr(resp, "parsed", None)
                if parsed is not None:
                    gives = parsed.gives_medical_advice
                    reason = parsed.reason
                else:
                    data = json.loads(resp.text)
                    gives = bool(data.get("gives_medical_advice", False))
                    reason = str(data.get("reason", ""))

                mark_ok(candidate_model)
                return GuardrailDecision(
                    blocked=gives,
                    category="clinical_interpretation" if gives else None,
                    reason=reason,
                    latency_ms=latency_ms,
                    error=None,
                    stage="output",
                    model=candidate_model,
                )
            except Exception as exc:
                last_exc = exc
                cooldown = get_cooldown_for_exception(exc)
                if cooldown:
                    mark_unavailable(candidate_model, cooldown[0], cooldown[1])
                    logger.warning(
                        "Gatekeeper output model '%s' hit cooldown (%s); trying fallback",
                        candidate_model,
                        cooldown[1],
                    )
                    continue
                else:
                    logger.error("Gatekeeper output model '%s' non-cooldown error: %s", candidate_model, exc)
                    raise

        if last_exc is not None:
            raise last_exc


# ── Singleton ────────────────────────────────────────────────────────────────

_guardrail: Optional[MedicalGuardrail] = None
_guardrail_lock = threading.Lock()


def get_medical_guardrail() -> MedicalGuardrail:
    """Return the process-wide singleton MedicalGuardrail."""
    global _guardrail
    if _guardrail is not None:
        return _guardrail
    with _guardrail_lock:
        if _guardrail is None:
            _guardrail = MedicalGuardrail()
        return _guardrail


def _reset_guardrail() -> None:
    """Reset the singleton — for tests only."""
    global _guardrail
    _guardrail = None
