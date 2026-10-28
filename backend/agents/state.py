"""
ReAct Agent State and Data Structures.

Defines the typed models for the Reasoning + Acting loop:
- ReActStep: Individual thought -> action -> observation cycle
- ReActResult: Complete output of an agent execution, fully compatible with orchestrator results
- ToolInvocation: Audit and logging payload for executed tools
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from llm.base import Message


@dataclass
class ToolInvocation:
    """One executed tool call, preserved for response and audit logging."""
    name: str
    arguments: Dict[str, Any]
    ok: bool
    summary: str
    record_ids: List[str] = field(default_factory=list)
    payload: Any = None
    execution_time_ms: float = 0.0


@dataclass
class ReActStep:
    """A single step in the ReAct (Reasoning + Acting) trace."""
    iteration: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    observation_summary: Optional[str] = None
    ok: bool = True
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation_summary": self.observation_summary,
            "ok": self.ok,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass
class ReActResult:
    """
    Complete result of a ReAct agent run.
    Drop-in compatible with OrchestratorResult while adding reasoning steps and timings.
    """
    content: str
    provider_used: str
    tokens_used: int = 0
    tool_calls: List[ToolInvocation] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    react_steps: List[ReActStep] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    error: Optional[str] = None


class ReActStepResponse(BaseModel):
    """Pydantic model for API responses exposing the agent's thought process."""
    iteration: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation_summary: Optional[str] = None
    ok: bool = True
    execution_time_ms: float = 0.0
