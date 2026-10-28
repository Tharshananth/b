"""Anthropic Claude LLM Provider Implementation"""
from typing import List, Optional, AsyncIterator
import json
import anthropic
from .base import BaseLLMProvider, Message, LLMResponse, ToolCall
from .tools import ToolSpec, to_anthropic
import logging

logger = logging.getLogger(__name__)

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider"""

    supports_tools = True

    
    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _format_for_tools(self, messages: List[Message]):
        """
        Message list in Anthropic wire format.

        Anthropic has no tool role: a tool result is a user message whose
        content is a tool_result block, and a tool request is an assistant
        message whose content mixes text and tool_use blocks.
        """
        out = []
        for m in messages:
            if m.role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }],
                })
            elif m.role == "assistant" and m.tool_calls:
                blocks = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def generate_response(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        **kwargs
    ) -> LLMResponse:
        try:
            formatted_messages = self._format_for_tools(messages)
            if tools:
                kwargs["tools"] = to_anthropic(tools)

            kwargs.setdefault("timeout", float(self.timeout))
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or "",
                messages=formatted_messages,
                **kwargs
            )
            
            text = "".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
            calls = [
                ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
                for b in response.content
                if getattr(b, "type", None) == "tool_use"
            ]

            return LLMResponse(
                content=text,
                model=self.model,
                provider="claude",
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason,
                tool_calls=calls or None,
            )
        except Exception as e:
            return self._handle_error(e, "generate_response")
    
    async def stream_response(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        try:
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            kwargs.setdefault("timeout", float(self.timeout))
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or "",
                messages=formatted_messages,
                **kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"

