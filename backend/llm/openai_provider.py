"""OpenAI LLM Provider Implementation"""
from typing import List, Optional, AsyncIterator
import json
import openai
from .base import BaseLLMProvider, Message, LLMResponse, ToolCall
from .tools import ToolSpec, to_openai
import logging

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""

    supports_tools = True

    
    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _format_for_tools(self, messages: List[Message], system_prompt):
        """Message list in OpenAI wire format, including tool turns."""
        out = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        for m in messages:
            if m.role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                })
            elif m.role == "assistant" and m.tool_calls:
                out.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                })
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
            formatted_messages = self._format_for_tools(messages, system_prompt)
            if tools:
                kwargs["tools"] = to_openai(tools)

            kwargs.setdefault("timeout", self.timeout)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            choice = response.choices[0]
            calls = None
            if getattr(choice.message, "tool_calls", None):
                calls = []
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    calls.append(ToolCall(id=tc.id, name=tc.function.name,
                                          arguments=args))

            return LLMResponse(
                content=choice.message.content or "",
                model=self.model,
                provider="openai",
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=choice.finish_reason,
                tool_calls=calls,
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
            formatted_messages = self.format_messages(messages, system_prompt)
            
            kwargs.setdefault("timeout", self.timeout)
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"

