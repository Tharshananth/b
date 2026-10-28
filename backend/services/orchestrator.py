"""
Orchestrator Service — ReAct Agent Facade.

Delegates execution to the production ReAct (Reasoning + Acting) Agent while
maintaining full backward compatibility with the existing router and audit pipeline.
"""
import logging
from typing import List, Optional

from llm.base import Message
from services.job_status import TenantScope
from agents.state import ToolInvocation, ReActResult as OrchestratorResult, ReActStep
from agents.react_agent import get_react_agent
from agents.tools.base import fence_tool_output as _fence

logger = logging.getLogger(__name__)

# Max iterations cap passed to the ReAct agent
MAX_ITERATIONS = 5


def run(
    user_message: str,
    scope: Optional[TenantScope] = None,
    history: Optional[List[Message]] = None,
    preferred_provider: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> OrchestratorResult:
    """
    Execute autonomous ReAct workflow for the incoming user message.

    Args:
        user_message: The user's query
        scope: Server-verified TenantScope
        history: Prior conversation turns
        preferred_provider: Optional LLM provider override
        max_iterations: Maximum thought-action iterations allowed
        session_id: Session identifier for logging
        user_id: User identifier for logging

    Returns:
        OrchestratorResult (ReActResult) with reasoning steps, sources, and audit calls.
    """
    agent = get_react_agent()
    return agent.run(
        user_message=user_message,
        scope=scope,
        history=history,
        preferred_provider=preferred_provider,
        max_iterations=max_iterations,
        session_id=session_id,
        user_id=user_id,
    )
