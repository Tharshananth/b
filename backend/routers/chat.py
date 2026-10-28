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
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from config import get_conversation_memory_config, get_security_config
from database import DataAccessLog, FeedbackInteraction, get_db
from llm.base import Message
from services import orchestrator, suggestions
from services.job_status import ScopeViolation, TenantScope

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ── Conversation memory ─────────────────────────────────────────────────────
# Stores the full Message list, tool turns included. A tool exchange is four
# messages (question, the model's tool request, the result, the answer) and all
# four have to survive into the next turn - otherwise a follow-up like "and why
# did that one fail?" reaches a model that has forgotten it ever ran a lookup.
#
# Still process-local, so it does not survive a restart and does not work
# across multiple workers. Redis is the eventual home.
session_memories: Dict[str, List[Message]] = {}


def _load_history(session_id: str) -> List[Message]:
    """Recent turns for a session, trimmed to the configured window."""
    history = session_memories.get(session_id, [])
    k = get_conversation_memory_config().buffer_window.k

    # Count complete exchanges from the end, keeping tool turns attached to the
    # user turn they belong to. Cutting between a tool request and its result
    # would leave a dangling call that every provider rejects.
    starts = [i for i, m in enumerate(history) if m.role == "user"]
    if len(starts) > k:
        return history[starts[-k]:]
    return history


def _save_history(session_id: str, messages: List[Message]) -> None:
    session_memories[session_id] = messages


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


# ── Endpoint ────────────────────────────────────────────────────────────────
@router.post("/", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Answer one message, using the knowledge base and job database as needed."""
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or "anonymous"
    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    logger.info("Chat | session=%s user=%s", session_id, user_id)

    scope = _resolve_scope(request)
    history = _load_history(session_id)

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
        # The database returned data outside the caller's scope. Never dress
        # this up as a normal error - fail hard and loudly.
        logger.critical("SCOPE VIOLATION on session %s: %s", session_id, e)
        raise HTTPException(
            status_code=500,
            detail="A data access check failed. This request was blocked.",
        )

    if result.error:
        # A failed turn is not committed to memory. result.messages would end
        # on the unanswered user message, so the next request would append a
        # second user turn back to back - a shape every provider rejects, which
        # would turn one upstream failure into a permanently broken session.
        logger.error("Orchestration error (turn not saved): %s", result.error)
    else:
        _save_history(session_id, result.messages)

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
            response=result.content,
            provider_used=result.provider_used,
            tokens_used=result.tokens_used,
        ))
        db.commit()
    except Exception as e:
        logger.error("Failed to save interaction: %s", e, exc_info=True)
        db.rollback()

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
        response=result.content,
        sources=[Source(**s) for s in result.sources],
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
        suggestions=suggestions.build(result.tool_calls),
        scoped=scope is not None,
    )


@router.get("/memory/info/{session_id}")
def get_memory_info(session_id: str):
    """Inspect what a session currently remembers, tool turns included."""
    memory_config = get_conversation_memory_config()
    history = session_memories.get(session_id, [])

    return {
        "session_id": session_id,
        "session_exists": session_id in session_memories,
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


@router.get("/history/{session_id}")
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


@router.delete("/history/{session_id}")
def delete_chat_history(session_id: str, db: Session = Depends(get_db)):
    """Delete a session's stored exchanges and its in-memory context."""
    try:
        session_memories.pop(session_id, None)
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


@router.post("/clear/{session_id}")
def clear_session_memory(session_id: str):
    """Forget a session's conversation without touching stored history."""
    existed = session_memories.pop(session_id, None) is not None
    return {
        "success": True,
        "message": (f"Cleared memory for session {session_id}" if existed
                    else f"No memory found for session {session_id}"),
    }
