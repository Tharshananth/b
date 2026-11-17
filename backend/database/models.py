"""Database models"""
from sqlalchemy import Boolean, Column, Float, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class FeedbackInteraction(Base):
    """
    Store chat interactions and feedback
    """
    __tablename__ = 'feedback_interactions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    message_id = Column(String, nullable=False, unique=True, index=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Interaction data
    question = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    provider_used = Column(String)
    tokens_used = Column(Integer)
    
    # Feedback data
    feedback_type = Column(String)
    feedback_comment = Column(Text)
    feedback_timestamp = Column(DateTime)
    
    def __repr__(self):
        return f"<FeedbackInteraction(id={self.id}, timestamp={self.timestamp})>"


class DataAccessLog(Base):
    """
    Record of every job-database read the chatbot performed.

    The interaction table stores what was said. This stores what was *seen* -
    whose records were returned, under whose scope, in answer to which message.
    For a clinical product that distinction matters: "the bot told a clinician
    about patient 2013" is a different question from "the bot read patient
    2013's record", and only the second one is auditable from here.

    Written on every tool call, successful or not. A refused lookup is as
    interesting to an auditor as a successful one.
    """
    __tablename__ = 'data_access_log'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Who asked
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    message_id = Column(String, nullable=False, index=True)

    # Under what scope the query ran. Null hospital means no identity was
    # resolved and the lookup was refused.
    scope_hospital = Column(String, index=True)
    scope_patient_id = Column(String, index=True)
    scope_source = Column(String)          # session | client | none

    # What ran
    tool_name = Column(String, nullable=False, index=True)
    arguments = Column(Text)               # JSON, as the model supplied them
    succeeded = Column(Boolean, nullable=False, default=True)
    failure_reason = Column(String)

    # What came back
    record_count = Column(Integer, default=0)
    record_ids = Column(Text)              # JSON list of job ids actually read

    def __repr__(self):
        return (f"<DataAccessLog({self.timestamp} {self.tool_name} "
                f"hospital={self.scope_hospital} n={self.record_count})>")


class AdminUser(Base):
    """Admin user account for backend administration"""
    __tablename__ = 'admin_users'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="admin")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<AdminUser(id={self.id}, username='{self.username}', role='{self.role}', active={self.is_active})>"


class AdminActionLog(Base):
    """Audit log of all admin operations"""
    __tablename__ = 'admin_action_log'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    username = Column(String, index=True, nullable=True)
    action = Column(String, index=True, nullable=False)
    target = Column(String, nullable=True)
    succeeded = Column(Boolean, default=True, nullable=False)
    detail = Column(Text, nullable=True)
    client_ip = Column(String, nullable=True)

    def __repr__(self):
        return f"<AdminActionLog({self.timestamp} username='{self.username}' action='{self.action}' succeeded={self.succeeded})>"



class GuardrailEvent(Base):
    """
    One row per guardrail check (input and optionally output).

    Does NOT store question or answer text — message_id links to
    FeedbackInteraction which already holds the full exchange.
    """
    __tablename__ = "guardrail_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Request identity — no PII content
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    message_id = Column(String, nullable=False, index=True)

    # Guardrail result — no question or answer text
    stage = Column(String, nullable=False)          # "input" | "output"
    category = Column(String, nullable=True)        # Category value or NULL on error
    decision = Column(String, nullable=False)       # "blocked" | "allowed" | "error"
    reason = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=False)
    model = Column(String, nullable=True)
    error = Column(Text, nullable=True)

    def __repr__(self):
        return (
            f"<GuardrailEvent({self.timestamp} stage={self.stage} "
            f"decision={self.decision} category={self.category})>"
        )


class RequestMetric(Base):
    """
    Per-request latency, cost, and routing metrics.
    Never stores message text.
    """
    __tablename__ = "request_metrics"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    message_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    path = Column(String, nullable=False, index=True)
    total_ms = Column(Float, nullable=False)
    guardrail_ms = Column(Float, nullable=True)
    agent_ms = Column(Float, nullable=True)
    llm_calls = Column(Integer, default=0, nullable=False)
    agent_rounds = Column(Integer, default=0, nullable=False)
    tokens = Column(Integer, nullable=True)
    cached_tokens = Column(Integer, nullable=True)
    provider = Column(String, nullable=True)
    model_used = Column(String, nullable=True)
    prompt_version = Column(String, nullable=True)
    parallel = Column(Boolean, default=False, nullable=False)

    def __repr__(self):
        return (
            f"<RequestMetric({self.timestamp} path='{self.path}' "
            f"total_ms={self.total_ms:.1f} llm_calls={self.llm_calls})>"
        )

