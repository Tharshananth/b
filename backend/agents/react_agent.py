"""
Production-grade ReAct (Reasoning + Acting) Agent Engine.

Implements the iterative Thought -> Action -> Observation -> Reflection cycle:
- Explicit scratchpad tracking each cognitive step
- Loop detection to prevent repetitive queries
- Server-side TenantScope enforcement with zero client-side bypass
- Comprehensive execution logging and timing metrics
- Resilient error handling and graceful fallbacks
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from config import get_config
from llm.base import Message, ToolCall
from llm.factory import get_llm_factory
from services.job_status import ScopeViolation, TenantScope
from agents.prompt import get_react_system_prompt
from agents.state import ReActResult, ReActStep, ToolInvocation
from agents.tools.registry import ToolRegistry, get_tool_registry

logger = logging.getLogger("agents.react_agent")

DEFAULT_MAX_ITERATIONS = 5


class ReActAgent:
    """
    Autonomous ReAct Agent for BrainSightAI VoxelBox Explore.

    Coordinates cognitive reasoning, tool execution, and reflection loops.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        max_iterations: Optional[int] = None,
    ):
        self.registry = registry or get_tool_registry()
        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            try:
                from config import get_agent_config
                self.max_iterations = get_agent_config().max_iterations
            except Exception:
                self.max_iterations = DEFAULT_MAX_ITERATIONS
        self.factory = get_llm_factory()

    def _extract_thought_and_content(self, text: str) -> Tuple[str, str]:
        """
        Parse explicit thought blocks or thoughts embedded in model output.
        Returns: (thought_str, cleaned_content_str)
        """
        if not text:
            return "", ""

        text_stripped = text.strip()

        # Handle <thought>...</thought> tags if model uses XML thinking tags
        if "<thought>" in text_stripped and "</thought>" in text_stripped:
            start = text_stripped.find("<thought>") + len("<thought>")
            end = text_stripped.find("</thought>")
            thought = text_stripped[start:end].strip()
            remainder = (text_stripped[:text_stripped.find("<thought>")] +
                         text_stripped[end + len("</thought>"):].strip())
            return thought, remainder

        # Handle canonical "Thought: ..." syntax
        if text_stripped.startswith("Thought:"):
            lines = text_stripped.splitlines()
            thought_lines = []
            rest_lines = []
            in_thought = True

            for line in lines:
                if in_thought:
                    if line.startswith("Action:") or line.startswith("Final Answer:"):
                        in_thought = False
                        rest_lines.append(line)
                    else:
                        thought_lines.append(line.replace("Thought:", "", 1).strip())
                else:
                    rest_lines.append(line)

            thought = " ".join(thought_lines).strip()
            content = "\n".join(rest_lines).strip()
            return thought, content

        return text_stripped, text_stripped

    def run(
        self,
        user_message: str,
        scope: Optional[TenantScope] = None,
        history: Optional[List[Message]] = None,
        preferred_provider: Optional[str] = None,
        max_iterations: Optional[int] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ReActResult:
        """
        Execute the ReAct loop for a single user turn.

        Args:
            user_message: Incoming user inquiry
            scope: Verified TenantScope (hospital, patient, etc.)
            history: Prior conversation turns
            preferred_provider: Optional model provider override
            max_iterations: Cap on tool calling turns
            session_id: Tracking ID for logs
            user_id: Caller ID for logs

        Returns:
            ReActResult containing final content, sources, steps, and audit logs.
        """
        start_overall = time.perf_counter()
        limit = max_iterations or self.max_iterations
        system_prompt = get_react_system_prompt()
        tool_specs = self.registry.get_all_tool_specs()

        logger.info(
            "ReActAgent.run() started | session=%s user=%s scope=%s max_iter=%d",
            session_id or "anonymous",
            user_id or "anonymous",
            f"hospital={scope.hospital}" if scope else "unscoped",
            limit,
        )
        logger.debug("User inquiry: %s", user_message[:160])

        original_messages: List[Message] = list(history or [])
        original_messages.append(Message(role="user", content=user_message))

        from llm.model_health import is_available, get_cooldown_for_exception

        failed_models: Set[str] = set()
        llm_calls_total: int = 0
        cached_tokens_total: int = 0

        while True:
            messages: List[Message] = list(original_messages)
            react_steps: List[ReActStep] = []
            invocations: List[ToolInvocation] = []
            sources: List[Dict[str, str]] = []
            tokens_accumulated = 0
            provider_used = "none"
            chosen_model: Optional[str] = None
            restarted = False

            # Used for loop detection
            seen_actions: Set[str] = set()

            for iteration in range(1, limit + 1):
                iter_start = time.perf_counter()
                logger.info("── ReAct Iteration %d/%d ──", iteration, limit)

                call_kwargs = {}
                if chosen_model is not None:
                    call_kwargs["model"] = chosen_model
                    call_kwargs["allow_model_fallback"] = False

                llm_calls_total += 1
                response = self.factory.generate_with_fallback(
                    messages=messages,
                    system_prompt=system_prompt,
                    preferred_provider=preferred_provider,
                    tools=tool_specs,
                    **call_kwargs,
                )

                provider_used = response.provider
                tokens_accumulated += response.tokens_used or 0
                cached_tokens_total += response.cached_tokens or 0

                # 1. Error from LLM provider
                if response.finish_reason == "error":
                    logger.error("LLM provider '%s' failed: %s", provider_used, response.error)

                    # Check if chosen_model failed mid-question with a cooldown-type error
                    # and restart whole question from scratch on the next available model
                    is_cooldown_error = (
                        (chosen_model is not None and not is_available(chosen_model))
                        or bool(get_cooldown_for_exception(Exception(response.error or "")))
                    )

                    if chosen_model is not None and is_cooldown_error:
                        failed_models.add(chosen_model)
                        gemini_provider = self.factory.get_provider("gemini")
                        remaining_candidates = []
                        if gemini_provider and hasattr(gemini_provider, "candidate_models"):
                            all_cands = gemini_provider.candidate_models()
                            remaining_candidates = [
                                m for m in all_cands
                                if m not in failed_models and is_available(m)
                            ]

                        if remaining_candidates:
                            logger.warning(
                                "Model '%s' failed mid-question (iteration %d) with cooldown error (%s). "
                                "Restarting question on next available model '%s'.",
                                chosen_model,
                                iteration,
                                response.error,
                                remaining_candidates[0],
                            )
                            restarted = True
                            break

                    elapsed_total = (time.perf_counter() - start_overall) * 1000.0
                    return ReActResult(
                        content=response.content or "An error occurred while communicating with the AI service.",
                        provider_used=provider_used,
                        tokens_used=tokens_accumulated,
                        tool_calls=invocations,
                        sources=sources,
                        messages=messages,
                        react_steps=react_steps,
                        total_execution_time_ms=elapsed_total,
                        error=response.error,
                        model_used=chosen_model or response.model,
                        capped=False,
                        llm_calls=llm_calls_total,
                        cached_tokens=cached_tokens_total,
                    )

                # Record chosen_model after the first successful round
                if chosen_model is None and response.model and response.model != "unknown":
                    chosen_model = response.model
                    logger.info("ReActAgent locked chosen model: %s", chosen_model)

                # 2. Model decides to finalize without more tools
                if not response.wants_tools:
                    thought, clean_answer = self._extract_thought_and_content(response.content)
                    step_duration = (time.perf_counter() - iter_start) * 1000.0

                    logger.info("Agent decided to answer (no tools requested).")
                    if thought and thought != clean_answer:
                        logger.info("Final Thought: %s", thought)

                    react_steps.append(
                        ReActStep(
                            iteration=iteration,
                            thought=thought or "Sufficient information available to answer.",
                            action=None,
                            action_input=None,
                            observation=None,
                            observation_summary="Completed final answer",
                            ok=True,
                            execution_time_ms=step_duration,
                        )
                    )

                    messages.append(Message(role="assistant", content=response.content))
                    elapsed_total = (time.perf_counter() - start_overall) * 1000.0

                    logger.info("ReActAgent completed using model: %s", chosen_model)
                    logger.info(
                        "ReActAgent completed in %.1fms across %d step(s)",
                        elapsed_total,
                        len(react_steps),
                    )

                    return ReActResult(
                        content=clean_answer or response.content,
                        provider_used=provider_used,
                        tokens_used=tokens_accumulated,
                        tool_calls=invocations,
                        sources=sources,
                        messages=messages,
                        react_steps=react_steps,
                        total_execution_time_ms=elapsed_total,
                        model_used=chosen_model or response.model,
                        capped=False,
                        llm_calls=llm_calls_total,
                        cached_tokens=cached_tokens_total,
                    )

                # 3. Model requested tools -> Action phase
                raw_thought, _ = self._extract_thought_and_content(response.content)
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                for call in response.tool_calls:
                    call_key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
                    is_duplicate = call_key in seen_actions
                    seen_actions.add(call_key)

                    thought_text = raw_thought
                    if not thought_text:
                        thought_text = f"Executing {call.name} to gather required context."

                    logger.info("Thought: %s", thought_text)
                    logger.info("Action: %s(%s)", call.name, call.arguments)

                    if is_duplicate:
                        logger.warning("Duplicate action detected: %s", call_key)

                    # Tool execution
                    tool = self.registry.get_tool(call.name)
                    if tool is None:
                        logger.warning("Agent requested unregistered tool: %s", call.name)
                        output = (
                            f"<{call.name}_result>\n"
                            f"{{\"error\": \"Tool '{call.name}' is not registered.\"}}\n"
                            f"</{call.name}_result>"
                        )
                        inv = ToolInvocation(
                            name=call.name,
                            arguments=call.arguments,
                            ok=False,
                            summary="unregistered tool",
                        )
                        found_sources = []
                    else:
                        output, inv, found_sources = tool.execute(call.arguments, scope)

                    invocations.append(inv)
                    sources.extend(found_sources)

                    # Append tool observation to message chain
                    messages.append(
                        Message(
                            role="tool",
                            name=call.name,
                            tool_call_id=call.id,
                            content=output,
                        )
                    )

                    step_duration = (time.perf_counter() - iter_start) * 1000.0
                    react_steps.append(
                        ReActStep(
                            iteration=iteration,
                            thought=thought_text,
                            action=call.name,
                            action_input=call.arguments,
                            observation=output[:500] if output else "",
                            observation_summary=inv.summary,
                            ok=inv.ok,
                            execution_time_ms=step_duration,
                        )
                    )

                    logger.info("Observation: %s (ok=%s)", inv.summary, inv.ok)

            if restarted:
                continue

            # 4. Iteration cap reached -> FORCED FINAL ANSWER
            logger.warning("ReActAgent reached iteration cap (%d); forcing final answer", limit)
            forced_request_messages = list(messages) + [
                Message(
                    role="user",
                    content=(
                        "Step limit reached. Answer the user's question now using "
                        "only the information gathered above. If something is still "
                        "unknown, say so briefly."
                    ),
                )
            ]

            final_kwargs = {}
            if chosen_model is not None:
                final_kwargs["model"] = chosen_model
                final_kwargs["allow_model_fallback"] = False

            llm_calls_total += 1
            final_start = time.perf_counter()
            final_response = self.factory.generate_with_fallback(
                messages=forced_request_messages,
                system_prompt=system_prompt,
                preferred_provider=preferred_provider,
                tools=tool_specs,
                tool_choice="none",
                **final_kwargs,
            )

            provider_used = final_response.provider
            tokens_accumulated += final_response.tokens_used or 0
            cached_tokens_total += final_response.cached_tokens or 0

            # Check if chosen_model hit cooldown error on final call
            if final_response.finish_reason == "error":
                is_cooldown_error = (
                    (chosen_model is not None and not is_available(chosen_model))
                    or bool(get_cooldown_for_exception(Exception(final_response.error or "")))
                )
                if chosen_model is not None and is_cooldown_error:
                    failed_models.add(chosen_model)
                    gemini_provider = self.factory.get_provider("gemini")
                    remaining_candidates = []
                    if gemini_provider and hasattr(gemini_provider, "candidate_models"):
                        all_cands = gemini_provider.candidate_models()
                        remaining_candidates = [
                            m for m in all_cands
                            if m not in failed_models and is_available(m)
                        ]
                    if remaining_candidates:
                        logger.warning(
                            "Model '%s' failed on forced final call with cooldown error (%s). "
                            "Restarting question on next available model '%s'.",
                            chosen_model,
                            final_response.error,
                            remaining_candidates[0],
                        )
                        continue

            if (
                final_response.finish_reason != "error"
                and final_response.content
                and not final_response.wants_tools
            ):
                thought, clean_text = self._extract_thought_and_content(final_response.content)
                final_duration = (time.perf_counter() - final_start) * 1000.0

                # Append assistant message to REAL messages (never the synthetic user message)
                messages.append(Message(role="assistant", content=final_response.content))
                react_steps.append(
                    ReActStep(
                        iteration=limit + 1,
                        thought=thought or "Step limit reached; answering with gathered context.",
                        action="final_answer (step limit)",
                        action_input=None,
                        observation=None,
                        observation_summary="Completed final answer (step limit)",
                        ok=True,
                        execution_time_ms=final_duration,
                    )
                )

                elapsed_total = (time.perf_counter() - start_overall) * 1000.0
                logger.info("ReActAgent forced final answer completed using model: %s", chosen_model)
                return ReActResult(
                    content=clean_text or final_response.content,
                    provider_used=provider_used,
                    tokens_used=tokens_accumulated,
                    tool_calls=invocations,
                    sources=sources,
                    messages=messages,
                    react_steps=react_steps,
                    total_execution_time_ms=elapsed_total,
                    error=None,
                    model_used=chosen_model or final_response.model,
                    capped=True,
                    llm_calls=llm_calls_total,
                    cached_tokens=cached_tokens_total,
                )

            # Fallback if final call failed otherwise or returned nothing
            elapsed_total = (time.perf_counter() - start_overall) * 1000.0
            fallback_msg = (
                "I investigated your inquiry across multiple steps, but reached the maximum "
                "lookup depth. Based on the information gathered, please review the steps above "
                "or contact support@brainsightai.com for further assistance."
            )
            logger.info("ReActAgent fallback completed using model: %s", chosen_model)
            return ReActResult(
                content=fallback_msg,
                provider_used=provider_used,
                tokens_used=tokens_accumulated,
                tool_calls=invocations,
                sources=sources,
                messages=messages,
                react_steps=react_steps,
                total_execution_time_ms=elapsed_total,
                error=f"Exceeded maximum iteration cap ({limit})",
                model_used=chosen_model,
                capped=True,
                llm_calls=llm_calls_total,
                cached_tokens=cached_tokens_total,
            )


# Singleton instance for simple imports
_agent = ReActAgent()


def get_react_agent() -> ReActAgent:
    """Get global ReAct agent instance."""
    return _agent
