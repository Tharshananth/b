"""
Google Gemini provider, on the google-genai SDK.

Replaces the old google-generativeai implementation, which is end-of-life and
flattened every message into one prompt string. That flattening made tool
calling impossible - the model could not distinguish a turn it had taken from
one the user had taken - so this builds structured Content turns instead.

Gemini's vocabulary differs from the rest of the codebase in two ways that this
module absorbs:
  - the assistant role is called "model"
  - function results are sent back in a *user* turn, not a dedicated tool role
  - function calls carry no id, so ids are synthesised from name and position
"""
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from google import genai
from google.genai import types

from .base import BaseLLMProvider, LLMResponse, Message, ToolCall
from .tools import ToolSpec, to_gemini

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider with tool-calling support."""

    supports_tools = True

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("Google API key is required")
        self.client = genai.Client(api_key=self.api_key)
        logger.info("Initialised Gemini model: %s", self.model)

    # ---------------- message conversion ----------------

    def _to_contents(self, messages: List[Message]) -> List[types.Content]:
        """Convert the provider-neutral message list into Gemini Content turns."""
        contents: List[types.Content] = []

        for msg in messages:
            if msg.role == "tool":
                # A tool result. Gemini expects the payload as structured data
                # in a user turn, keyed by the function's name.
                try:
                    payload = json.loads(msg.content) if msg.content else {}
                except json.JSONDecodeError:
                    payload = {"result": msg.content}
                if not isinstance(payload, dict):
                    payload = {"result": payload}
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.name or "unknown",
                                response=payload,
                            )
                        ],
                    )
                )

            elif msg.role == "assistant" and msg.tool_calls:
                # The model's own request for tools, replayed so it can see
                # what it already asked for.
                parts = []
                for tc in msg.tool_calls:
                    part = types.Part.from_function_call(
                        name=tc.name, args=tc.arguments
                    )
                    signature = (tc.provider_data or {}).get("thought_signature")
                    if signature is not None:
                        part.thought_signature = signature
                    parts.append(part)
                contents.append(types.Content(role="model", parts=parts))

            elif msg.content:
                contents.append(
                    types.Content(
                        role="model" if msg.role == "assistant" else "user",
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        return contents

    def _config(
        self,
        system_prompt: Optional[str],
        tools: Optional[List[ToolSpec]],
    ) -> types.GenerateContentConfig:
        kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "top_p": self.top_p,
            # HttpOptions.timeout is milliseconds; self.timeout is seconds.
            "http_options": types.HttpOptions(
                timeout=int(self.timeout * 1000),
                # Gemini returns 503 "high demand" in short bursts that clear
                # in a second or two, so those are retried in-provider rather
                # than failed over.
                #
                # 429 is deliberately NOT retried. It means the key is out of
                # quota, and retrying spends more of it while the user waits.
                # It should surface immediately so the caller can say the
                # service is busy - and so the real fix, a paid tier, stays
                # visible instead of being hidden behind silent backoff.
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=1.0,
                    max_delay=8.0,
                    exp_base=2.0,
                    http_status_codes=[500, 502, 503, 504],
                ),
            ),
        }
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        if tools:
            kwargs["tools"] = to_gemini(tools)
            # The SDK will happily execute Python callables on its own. Tools
            # here are declarations only, and execution stays server-side where
            # the tenant scope is injected - so auto-calling is switched off
            # explicitly rather than relied upon to stay inactive.
            kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _parse(response) -> tuple[str, List[ToolCall]]:
        """Pull text and function calls out of a response, defensively."""
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            for i, part in enumerate(getattr(content, "parts", None) or []):
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append(
                        ToolCall(
                            # Gemini supplies no call id; position is stable
                            # within one response and that is all we need.
                            id=f"{fc.name}_{i}",
                            name=fc.name,
                            arguments=dict(fc.args or {}),
                            # Gemini 3 signs each function call. Replaying the
                            # call without this signature is rejected with
                            # "Function call is missing a thought_signature",
                            # so it is carried through the neutral ToolCall and
                            # restored in _to_contents.
                            provider_data={
                                "thought_signature": getattr(
                                    part, "thought_signature", None
                                )
                            },
                        )
                    )
                    continue
                if getattr(part, "text", None):
                    text_parts.append(part.text)

        return "".join(text_parts), tool_calls

    # ---------------- generation ----------------

    def generate_response(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        **kwargs,
    ) -> LLMResponse:
        try:
            contents = self._to_contents(messages)
            logger.debug(
                "Gemini request: %d turns, tools=%s",
                len(contents),
                [t.name for t in tools] if tools else None,
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._config(system_prompt, tools),
            )

            text, tool_calls = self._parse(response)

            usage = getattr(response, "usage_metadata", None)
            tokens = getattr(usage, "total_token_count", None) if usage else None

            if tool_calls:
                logger.info(
                    "Gemini requested %d tool call(s): %s",
                    len(tool_calls),
                    [tc.name for tc in tool_calls],
                )
                return LLMResponse(
                    content=text,
                    model=self.model,
                    provider="gemini",
                    tokens_used=tokens,
                    finish_reason="tool_calls",
                    tool_calls=tool_calls,
                )

            if text:
                return LLMResponse(
                    content=text,
                    model=self.model,
                    provider="gemini",
                    tokens_used=tokens,
                    finish_reason="stop",
                )

            # No text and no tool call usually means a safety block or an empty
            # candidate list; surface it as an error so fallback can take over.
            reason = None
            if getattr(response, "candidates", None):
                reason = getattr(response.candidates[0], "finish_reason", None)
            logger.warning("Empty Gemini response (finish_reason=%s)", reason)
            return LLMResponse(
                content="I couldn't generate a response. Please try again.",
                model=self.model,
                provider="gemini",
                finish_reason="error",
                error=f"Empty response (finish_reason={reason})",
            )

        except Exception as e:
            logger.error(f"Gemini error: {e}", exc_info=True)
            return self._handle_error(e, "generate_response")

    async def stream_response(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        try:
            contents = self._to_contents(messages)
            logger.debug("Gemini streaming request: %d turns", len(contents))

            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self._config(system_prompt, None),
            )

            chunks = 0
            for chunk in stream:
                text, _ = self._parse(chunk)
                if text:
                    chunks += 1
                    yield text

            if chunks == 0:
                logger.warning("Gemini stream produced no chunks")
                yield "I couldn't generate a response. Please try again."

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield "Error: I encountered an issue. Please try again."
