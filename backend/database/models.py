"""Database models"""
from sqlalchemy import Boolean, Column, String, Text, DateTime, Integer
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
