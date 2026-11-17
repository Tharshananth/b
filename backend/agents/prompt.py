"""
ReAct Agent Prompt Templates and Guidance.

Combines BrainSightAI domain identity with the ReAct (Reasoning + Acting)
cognitive framework — without leaking internal labels into user-visible replies.
"""
from typing import Optional
from config import get_prompts_config
from prompts.loader import load_prompt


def get_react_system_prompt(custom_instructions: Optional[str] = None) -> str:
    """
    Build complete system prompt combining base system prompt from config
    with ReAct agent guidelines.
    """
    cfg = get_prompts_config()
    base_prompt = load_prompt(cfg.system)
    react_instructions = load_prompt(cfg.react_instructions)
    parts = [base_prompt.strip(), react_instructions.strip()]
    if custom_instructions:
        parts.append(custom_instructions.strip())
    return "\n\n" + "\n\n".join(parts) + "\n"
