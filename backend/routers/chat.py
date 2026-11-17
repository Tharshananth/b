"""
Chat endpoint.

One tool-calling orchestrator handles every message. The model decides whether
a question needs the knowledge base, the job database, both, or neither - there
is no router deciding that up front.

The endpoint is deliberately a plain `def`, not `async def`: the LLM and Mongo
calls beneath it are blocking, so FastAPI runs this in a threadpool instead of
stalling the event loop for every other request in flight.
"""
import json
import logging
import uuid
import concurrent.futures
import time
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from config import (
    get_conversation_memory_config,
    get_security_config,
    get_guardrails_config,
    get_citations_config,
    get_fast_path_config,
    get_performance_config,
)
from database import DataAccessLog, FeedbackInteraction, GuardrailEvent, RequestMetric, SessionLocal, get_db
from fast_path import match_greeting
from llm.base import Message
from auth.dependencies import require_admin
from memory import get_memory_store, trim
from guardrails import get_medical_guardrail, GuardrailDecision
from services import orchestrator, suggestions
from services.citations import verify_citations
from services.job_status import ScopeViolation, TenantScope

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])

_chat_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="chat_worker",
)


# ── Conversation memory ─────────────────────────────────────────────────────
def _load_history(session_id: str) -> List[Message]:
    """Recent turns for a session, trimmed to the configured window."""
    k = get_conversation_memory_config().buffer_window.k
    history = get_memory_store().load(session_id)
    return trim(history, k)


def _save_history(session_id: str, messages: List[Message]) -> None:
    get_memory_store().save(session_id, messages)


# ── Scope resolution ────────────────────────────────────────────────────────
def _resolve_scope(request: "ChatRequest") -> Optional[TenantScope]:
    """
    Work out whose data this request may see.

    Order of preference, once the login handshake exists, will be:
      1. the authenticated session   (not built yet - frontend team)
      2. the request body            (testing only, behind allow_client_scope)
      3. nothing                     (job lookups refuse)

    Step 2 is self-declared and therefore not authentication. It exists so the
    flow can be demonstrated before step 1 lands, and the config flag defaults
    to off so it cannot reach production by accident.
    """
    cfg = get_security_config()

    if not cfg.allow_client_scope:
        # Production path. Until sessions carry identity there is no scope, and
        # job lookups will refuse rather than return anything unscoped.
        return None

    hospital = request.hospital or cfg.default_hospital
    if not hospital:
        return None

    logger.warning(
        "Using CLIENT-SUPPLIED scope (hospital=%s patient=%s). "
        "This is self-declared, not authenticated - testing only.",
        hospital, request.patient_id,
    )
    return TenantScope(hospital=hospital, patient_id=request.patient_id)


# ── Models ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    provider: Optional[str] = None
    user_id: Optional[str] = None

    # Testing-only scope. Ignored unless security.allow_client_scope is true.
    hospital: Optional[str] = None
    patient_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("message cannot be blank")
        return s


class Source(BaseModel):
    title: str
    url: str
    content: str
    page: Optional[int] = None


class ToolUse(BaseModel):
    name: str
    arguments: dict
    ok: bool
    summary: str


class ReActStepModel(BaseModel):
    iteration: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation_summary: Optional[str] = None
    ok: bool = True
    execution_time_ms: float = 0.0


class ChatResponse(BaseModel):
    response: str
    sources: List[Source]
    session_id: str
    message_id: str
    success: bool = True
    provider_used: str
    tokens_used: Optional[int] = None
    tools_used: List[ToolUse] = []
    react_steps: List[ReActStepModel] = []
    suggestions: List[str] = []
    scoped: bool = False
    guardrail: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    session_id: Optional[str] = None
    timestamp: str


def _write_audit(db: Session, *, result, scope: Optional[TenantScope],
                 user_id: str, session_id: str, message_id: str) -> None:
    """
    Record every tool call against the job database.

    Best-effort: an audit failure is logged loudly but never costs the user
    their answer. If audit becomes a hard compliance requirement, this is the
    line to change - write first, then answer.
    """
    source = "none"
    if scope is not None:
        source = "client" if get_security_config().allow_client_scope else "session"

    try:
        for call in result.tool_calls:
            if call.name != "get_job_status":
                continue   # knowledge base reads touch no patient data
            db.add(DataAccessLog(
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                scope_hospital=scope.hospital if scope else None,
                scope_patient_id=scope.patient_id if scope else None,
                scope_source=source,
                tool_name=call.name,
                arguments=json.dumps(call.arguments, default=str),
                succeeded=call.ok,
                failure_reason=None if call.ok else call.summary,
                record_count=len(call.record_ids),
                record_ids=json.dumps(call.record_ids),
            ))
        db.commit()
    except Exception as e:
        logger.error("AUDIT WRITE FAILED: %s", e, exc_info=True)
        db.rollback()


# ── Guardrail event logger ───────────────────────────────────────────────────
def _log_guardrail_event(
    db: Session,
    *,
    session_id: str,
    user_id: str,
    message_id: str,
    decision: "GuardrailDecision",
    model: str,
) -> None:
    """Write one GuardrailEvent row best-effort. Never raises."""
    try:
        if decision.error:
            decision_str = "error"
        elif decision.blocked:
            decision_str = "blocked"
        else:
            decision_str = "allowed"
        db.add(GuardrailEvent(
            session_id=session_id,
            user_id=user_id,
            message_id=message_id,
            stage=decision.stage,
            category=decision.category,
            decision=decision_str,
            reason=decision.reason,
            latency_ms=decision.latency_ms,
            model=model,
            error=decision.error,
        ))
        db.commit()
    except Exception as exc:
        logger.error("GuardrailEvent write failed: %s", exc)
        db.rollback()


# ── Background agent callback (parallel mode) ────────────────────────────────
def _attach_background_agent_callback(
    fut,
    *,
    scope: Optional[TenantScope],
    user_id: str,
    session_id: str,
    message_id: str,
) -> None:
    """
    Handle background agent completion when guardrail blocked or routed.
    Opens its own DB session, writes audit, logs CRITICAL on ScopeViolation, never raises.
    """
    def _done_cb(f):
        db_bg = SessionLocal()
        try:
            try:
                res = f.result()
                _write_audit(
                    db_bg,
                    result=res,
                    scope=scope,
                    user_id=user_id,
                    session_id=session_id,
                    message_id=message_id,
                )
            except ScopeViolation as sve:
                logger.critical("SCOPE VIOLATION on session %s: %s", session_id, sve)
            except Exception as bg_e:
                logger.error("Background agent execution failed: %s", bg_e, exc_info=True)
        except Exception as outer_e:
            logger.error("Error in background agent callback: %s", outer_e)
        finally:
            db_bg.close()

    fut.add_done_callback(_done_cb)


# ── Request metric logger ───────────────────────────────────────────────────
def _write_metric(
    db: Session,
    *,
    message_id: str,
    session_id: str,
    path: str,
    total_ms: float,
    guardrail_ms: Optional[float] = None,
    agent_ms: Optional[float] = None,
    llm_calls: int = 0,
    agent_rounds: int = 0,
    tokens: Optional[int] = None,
    cached_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    model_used: Optional[str] = None,
    parallel: bool = False,
) -> None:
    """Write exactly one RequestMetric row best-effort. Never raises."""
    p_ver = None
    try:
        from prompts.loader import prompt_version
        p_ver = prompt_version()
    except Exception:
        p_ver = None

    try:
        metric = RequestMetric(
            message_id=message_id,
            session_id=session_id,
            path=path,
            total_ms=round(total_ms, 2),
            guardrail_ms=round(guardrail_ms, 2) if guardrail_ms is not None else None,
            agent_ms=round(agent_ms, 2) if agent_ms is not None else None,
            llm_calls=llm_calls,
            agent_rounds=agent_rounds,
            tokens=tokens,
            cached_tokens=cached_tokens,
            provider=provider,
            model_used=model_used,
            prompt_version=p_ver,
            parallel=parallel,
        )
        db.add(metric)
        db.commit()
    except Exception as exc:
        logger.error("Failed to write RequestMetric: %s", exc)
        db.rollback()


# ── Endpoint ────────────────────────────────────────────────────────────────
@router.post("/", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Answer one message, using the knowledge base and job database as needed."""
    start_time = time.perf_counter()
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or "anonymous"
    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    logger.info("Chat | session=%s user=%s", session_id, user_id)

    scope = _resolve_scope(request)
    history = _load_history(session_id)

    # ── Step 1: Fast-path instant greetings (no LLM, no guardrail) ───────────
    fast_path_cfg = get_fast_path_config()
    if fast_path_cfg.enabled:
        kind = match_greeting(request.message)
        if kind:
            reply_text = getattr(fast_path_cfg.replies, kind, fast_path_cfg.replies.greeting)
            _save_history(
                session_id,
                history + [
                    Message(role="user", content=request.message),
                    Message(role="assistant", content=reply_text),
                ],
            )
            try:
                db.add(FeedbackInteraction(
                    user_id=user_id,
                    session_id=session_id,
                    message_id=message_id,
                    timestamp=datetime.utcnow(),
                    question=request.message,
                    response=reply_text,
                    provider_used="fast_path",
                    tokens_used=None,
                ))
                db.commit()
            except Exception as exc:
                logger.error("Failed to save fast_path interaction: %s", exc, exc_info=True)
                db.rollback()

            total_ms = (time.perf_counter() - start_time) * 1000.0
            _write_metric(
                db,
                message_id=message_id,
                session_id=session_id,
                path="fast_path",
                total_ms=total_ms,
                guardrail_ms=None,
                agent_ms=None,
                llm_calls=0,
                agent_rounds=0,
                tokens=None,
                cached_tokens=None,
                provider="fast_path",
                model_used=None,
                parallel=False,
            )

            return ChatResponse(
                response=reply_text,
                sources=[],
                session_id=session_id,
                message_id=message_id,
                success=True,
                provider_used="fast_path",
                tokens_used=None,
                tools_used=[],
                react_steps=[],
                suggestions=[],
                scoped=scope is not None,
                guardrail=None,
            )

    gr_cfg = get_guardrails_config().medical
    is_parallel = bool(get_performance_config().parallel_gatekeeper and gr_cfg.enabled)
    input_decision: Optional[GuardrailDecision] = None
    result = None

    if is_parallel:
        # Submit both guardrail and agent concurrently
        guardrail_future = _chat_executor.submit(
            get_medical_guardrail().check_input, request.message, history
        )
        agent_future = _chat_executor.submit(
            orchestrator.run,
            user_message=request.message,
            scope=scope,
            history=history,
            preferred_provider=request.provider,
            session_id=session_id,
            user_id=user_id,
        )

        input_decision = guardrail_future.result()
        _log_guardrail_event(
            db,
            session_id=session_id,
            user_id=user_id,
            message_id=message_id,
            decision=input_decision,
            model=input_decision.model or gr_cfg.model,
        )

        if input_decision.blocked:
            canned = gr_cfg.canned_response
            _save_history(
                session_id,
                history + [
                    Message(role="user", content=request.message),
                    Message(role="assistant", content=canned),
                ],
            )
            try:
                db.add(FeedbackInteraction(
                    user_id=user_id,
                    session_id=session_id,
                    message_id=message_id,
                    timestamp=datetime.utcnow(),
                    question=request.message,
                    response=canned,
                    provider_used="guardrail",
                    tokens_used=None,
                ))
                db.commit()
            except Exception as exc:
                logger.error("Failed to save guardrail interaction: %s", exc, exc_info=True)
                db.rollback()

            _attach_background_agent_callback(
                agent_future,
                scope=scope,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
            )

            total_ms = (time.perf_counter() - start_time) * 1000.0
            _write_metric(
                db,
                message_id=message_id,
                session_id=session_id,
                path="guardrail_block",
                total_ms=total_ms,
                guardrail_ms=input_decision.latency_ms,
                agent_ms=None,
                llm_calls=1,
                agent_rounds=0,
                tokens=None,
                cached_tokens=None,
                provider="guardrail",
                model_used=input_decision.model or gr_cfg.model,
                parallel=True,
            )

            return ChatResponse(
                response=canned,
                sources=[],
                session_id=session_id,
                message_id=message_id,
                success=True,
                provider_used="guardrail",
                tokens_used=None,
                tools_used=[],
                react_steps=[],
                suggestions=[],
                scoped=scope is not None,
                guardrail="medical_advice",
            )

        if input_decision.category in ("greeting", "out_of_scope"):
            reply_text = getattr(
                fast_path_cfg.replies, input_decision.category, fast_path_cfg.replies.greeting
            )
            _save_history(
                session_id,
                history + [
                    Message(role="user", content=request.message),
                    Message(role="assistant", content=reply_text),
                ],
            )
            try:
                db.add(FeedbackInteraction(
                    user_id=user_id,
                    session_id=session_id,
                    message_id=message_id,
                    timestamp=datetime.utcnow(),
                    question=request.message,
                    response=reply_text,
                    provider_used="guardrail_route",
                    tokens_used=None,
                ))
                db.commit()
            except Exception as exc:
                logger.error("Failed to save guardrail route interaction: %s", exc, exc_info=True)
                db.rollback()

            _attach_background_agent_callback(
                agent_future,
                scope=scope,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
            )

            total_ms = (time.perf_counter() - start_time) * 1000.0
            _write_metric(
                db,
                message_id=message_id,
                session_id=session_id,
                path="guardrail_route",
                total_ms=total_ms,
                guardrail_ms=input_decision.latency_ms,
                agent_ms=None,
                llm_calls=1,
                agent_rounds=0,
                tokens=None,
                cached_tokens=None,
                provider="guardrail_route",
                model_used=input_decision.model or gr_cfg.model,
                parallel=True,
            )

            return ChatResponse(
                response=reply_text,
                sources=[],
                session_id=session_id,
                message_id=message_id,
                success=True,
                provider_used="guardrail_route",
                tokens_used=None,
                tools_used=[],
                react_steps=[],
                suggestions=[],
                scoped=scope is not None,
                guardrail=None,
            )

        # Allowed: wait for agent
        try:
            result = agent_future.result()
        except ScopeViolation as e:
            logger.critical("SCOPE VIOLATION on session %s: %s", session_id, e)
            total_ms = (time.perf_counter() - start_time) * 1000.0
            _write_metric(
                db,
                message_id=message_id,
                session_id=session_id,
                path="agent_error",
                total_ms=total_ms,
                guardrail_ms=input_decision.latency_ms if input_decision else None,
                agent_ms=None,
                llm_calls=1,
                agent_rounds=0,
                tokens=None,
                cached_tokens=None,
                provider="none",
                model_used=None,
                parallel=True,
            )
            raise HTTPException(
                status_code=500,
                detail="A data access check failed. This request was blocked.",
            )

    else:
        # Sequential path: check input first
        if gr_cfg.enabled:
            input_decision = get_medical_guardrail().check_input(request.message, history)
            _log_guardrail_event(
                db,
                session_id=session_id,
                user_id=user_id,
                message_id=message_id,
                decision=input_decision,
                model=input_decision.model or gr_cfg.model,
            )
            if input_decision.blocked:
                canned = gr_cfg.canned_response
                _save_history(
                    session_id,
                    history + [
                        Message(role="user", content=request.message),
                        Message(role="assistant", content=canned),
                    ],
                )
                try:
                    db.add(FeedbackInteraction(
                        user_id=user_id,
                        session_id=session_id,
                        message_id=message_id,
                        timestamp=datetime.utcnow(),
                        question=request.message,
                        response=canned,
                        provider_used="guardrail",
                        tokens_used=None,
                    ))
                    db.commit()
                except Exception as exc:
                    logger.error("Failed to save guardrail interaction: %s", exc, exc_info=True)
                    db.rollback()

                total_ms = (time.perf_counter() - start_time) * 1000.0
                _write_metric(
                    db,
                    message_id=message_id,
                    session_id=session_id,
                    path="guardrail_block",
                    total_ms=total_ms,
                    guardrail_ms=input_decision.latency_ms,
                    agent_ms=None,
                    llm_calls=1,
                    agent_rounds=0,
                    tokens=None,
                    cached_tokens=None,
                    provider="guardrail",
                    model_used=input_decision.model or gr_cfg.model,
                    parallel=False,
                )

                return ChatResponse(
                    response=canned,
                    sources=[],
                    session_id=session_id,
                    message_id=message_id,
                    success=True,
                    provider_used="guardrail",
                    tokens_used=None,
                    tools_used=[],
                    react_steps=[],
                    suggestions=[],
                    scoped=scope is not None,
                    guardrail="medical_advice",
                )

            # Step 4: Let gatekeeper route greetings and out_of_scope
            if input_decision.category in ("greeting", "out_of_scope"):
                reply_text = getattr(
                    fast_path_cfg.replies, input_decision.category, fast_path_cfg.replies.greeting
                )
                _save_history(
                    session_id,
                    history + [
                        Message(role="user", content=request.message),
                        Message(role="assistant", content=reply_text),
                    ],
                )
                try:
                    db.add(FeedbackInteraction(
                        user_id=user_id,
                        session_id=session_id,
                        message_id=message_id,
                        timestamp=datetime.utcnow(),
                        question=request.message,
                        response=reply_text,
                        provider_used="guardrail_route",
                        tokens_used=None,
                    ))
                    db.commit()
                except Exception as exc:
                    logger.error("Failed to save guardrail route interaction: %s", exc, exc_info=True)
                    db.rollback()

                total_ms = (time.perf_counter() - start_time) * 1000.0
                _write_metric(
                    db,
                    message_id=message_id,
                    session_id=session_id,
                    path="guardrail_route",
                    total_ms=total_ms,
                    guardrail_ms=input_decision.latency_ms,
                    agent_ms=None,
                    llm_calls=1,
                    agent_rounds=0,
                    tokens=None,
                    cached_tokens=None,
                    provider="guardrail_route",
                    model_used=input_decision.model or gr_cfg.model,
                    parallel=False,
                )

                return ChatResponse(
                    response=reply_text,
                    sources=[],
                    session_id=session_id,
                    message_id=message_id,
                    success=True,
                    provider_used="guardrail_route",
                    tokens_used=None,
                    tools_used=[],
                    react_steps=[],
                    suggestions=[],
                    scoped=scope is not None,
                    guardrail=None,
                )

        try:
            result = orchestrator.run(
                user_message=request.message,
                scope=scope,
                history=history,
                preferred_provider=request.provider,
                session_id=session_id,
                user_id=user_id,
            )
        except ScopeViolation as e:
            logger.critical("SCOPE VIOLATION on session %s: %s", session_id, e)
            total_ms = (time.perf_counter() - start_time) * 1000.0
            _write_metric(
                db,
                message_id=message_id,
                session_id=session_id,
                path="agent_error",
                total_ms=total_ms,
                guardrail_ms=input_decision.latency_ms if input_decision else None,
                agent_ms=None,
                llm_calls=(1 if input_decision else 0),
                agent_rounds=0,
                tokens=None,
                cached_tokens=None,
                provider="none",
                model_used=None,
                parallel=False,
            )
            raise HTTPException(
                status_code=500,
                detail="A data access check failed. This request was blocked.",
            )

    # ── Citations verification (before memory save & output check) ───────────
    if not result.error and get_citations_config().verify:
        check = verify_citations(result.content, result.sources, message_id=message_id)
        if check.invalid > 0:
            logger.warning(
                "Citations verified | msg=%s valid=%d invalid=%d missing_page=%d",
                message_id, check.valid, check.invalid, check.missing_page,
            )
        else:
            logger.info(
                "Citations verified | msg=%s valid=%d invalid=%d missing_page=%d",
                message_id, check.valid, check.invalid, check.missing_page,
            )
        result.content = check.text
        if result.messages:
            for m in reversed(result.messages):
                if m.role == "assistant":
                    m.content = check.text
                    break

    if result.error:
        logger.error("Orchestration error (turn not saved): %s", result.error)
    else:
        _save_history(session_id, result.messages)

    # ── Medical guardrail output check (optional) ────────────────────────────
    guardrail_field: Optional[str] = None
    response_content = result.content
    response_sources = result.sources
    response_suggestions_flag = True
    out_decision = None
    if not result.error and gr_cfg.enabled and gr_cfg.output_check:
        out_decision = get_medical_guardrail().check_output(result.content)
        _log_guardrail_event(
            db,
            session_id=session_id,
            user_id=user_id,
            message_id=message_id,
            decision=out_decision,
            model=out_decision.model or gr_cfg.model,
        )
        if out_decision.blocked:
            canned = gr_cfg.canned_response
            _save_history(
                session_id,
                history + [
                    Message(role="user", content=request.message),
                    Message(role="assistant", content=canned),
                ],
            )
            response_content = canned
            response_sources = []
            response_suggestions_flag = False
            guardrail_field = "medical_advice"

    _write_audit(db, result=result, scope=scope, user_id=user_id,
                 session_id=session_id, message_id=message_id)

    # Persist the exchange. A failure here must not lose the user's answer.
    try:
        db.add(FeedbackInteraction(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            timestamp=datetime.utcnow(),
            question=request.message,
            response=response_content,
            provider_used=result.provider_used,
            tokens_used=result.tokens_used,
        ))
        db.commit()
    except Exception as e:
        logger.error("Failed to save interaction: %s", e, exc_info=True)
        db.rollback()

    # Determine metric path:
    if result.error:
        metric_path = "agent_error"
    elif getattr(result, "capped", False):
        metric_path = "agent_capped"
    else:
        metric_path = "agent"

    total_ms = (time.perf_counter() - start_time) * 1000.0
    guardrail_ms = None
    if input_decision:
        guardrail_ms = input_decision.latency_ms
    if out_decision:
        guardrail_ms = (guardrail_ms or 0.0) + out_decision.latency_ms

    llm_calls = (
        (1 if input_decision else 0)
        + (1 if out_decision else 0)
        + getattr(result, "llm_calls", 0)
    )

    _write_metric(
        db,
        message_id=message_id,
        session_id=session_id,
        path=metric_path,
        total_ms=total_ms,
        guardrail_ms=guardrail_ms,
        agent_ms=getattr(result, "total_execution_time_ms", None),
        llm_calls=llm_calls,
        agent_rounds=len(getattr(result, "react_steps", []) or []),
        tokens=result.tokens_used,
        cached_tokens=getattr(result, "cached_tokens", None),
        provider=result.provider_used,
        model_used=getattr(result, "model_used", None),
        parallel=is_parallel,
    )

    raw_steps = getattr(result, "react_steps", []) or []
    step_models = [
        ReActStepModel(
            iteration=s.iteration,
            thought=s.thought,
            action=s.action,
            action_input=s.action_input,
            observation_summary=s.observation_summary,
            ok=s.ok,
            execution_time_ms=s.execution_time_ms,
        )
        for s in raw_steps
    ]

    return ChatResponse(
        response=response_content,
        sources=[Source(**s) for s in response_sources],
        session_id=session_id,
        message_id=message_id,
        success=not result.error,
        provider_used=result.provider_used,
        tokens_used=result.tokens_used,
        tools_used=[
            ToolUse(name=t.name, arguments=t.arguments, ok=t.ok, summary=t.summary)
            for t in result.tool_calls
        ],
        react_steps=step_models,
        suggestions=suggestions.build(result.tool_calls) if response_suggestions_flag else [],
        scoped=scope is not None,
        guardrail=guardrail_field,
    )


@router.get("/memory/info/{session_id}", dependencies=[Depends(require_admin)])
def get_memory_info(session_id: str):
    """Inspect what a session currently remembers, tool turns included."""
    memory_config = get_conversation_memory_config()
    store = get_memory_store()
    history = store.load(session_id)
    exists = store.exists(session_id)

    return {
        "session_id": session_id,
        "session_exists": exists,
        "backend": store.status()["backend"],
        "config": {
            "type": memory_config.type,
            "buffer_window_k": memory_config.buffer_window.k,
            "max_tokens": memory_config.buffer_window.max_tokens,
        },
        "message_count": len(history),
        "messages": [
            {
                "role": m.role,
                "tool_calls": [tc.name for tc in (m.tool_calls or [])] or None,
                "tool_name": m.name,
                "content": (m.content[:100] + "...") if len(m.content) > 100 else m.content,
            }
            for m in history
        ],
    }


@router.get("/history/{session_id}", dependencies=[Depends(require_admin)])
def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """Stored exchanges for a session."""
    try:
        rows = (
            db.query(FeedbackInteraction)
            .filter(FeedbackInteraction.session_id == session_id)
            .order_by(FeedbackInteraction.timestamp.asc())
            .all()
        )
        return {
            "session_id": session_id,
            "message_count": len(rows),
            "history": [
                {
                    "message_id": r.message_id,
                    "timestamp": r.timestamp.isoformat(),
                    "question": r.question,
                    "response": r.response,
                    "provider": r.provider_used,
                    "feedback_type": r.feedback_type,
                }
                for r in rows
            ],
        }
    except Exception as e:
        logger.error("Failed to get history: %s", e)
        raise HTTPException(status_code=500, detail="Could not load history")


@router.delete("/history/{session_id}", dependencies=[Depends(require_admin)])
def delete_chat_history(session_id: str, db: Session = Depends(get_db)):
    """Delete a session's stored exchanges and its in-memory context."""
    try:
        get_memory_store().clear(session_id)
        deleted = (
            db.query(FeedbackInteraction)
            .filter(FeedbackInteraction.session_id == session_id)
            .delete()
        )
        db.commit()
        return {"success": True,
                "message": f"Deleted {deleted} messages from session {session_id}"}
    except Exception as e:
        logger.error("Failed to delete history: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not delete history")


@router.post("/clear/{session_id}", dependencies=[Depends(require_admin)])
def clear_session_memory(session_id: str):
    """Forget a session's conversation without touching stored history."""
    existed = get_memory_store().clear(session_id)
    return {
        "success": True,
        "message": (f"Cleared memory for session {session_id}" if existed
                    else f"No memory found for session {session_id}"),
    }
