"""HuggingFace LLM Provider Implementation"""
from typing import List, Optional, AsyncIterator
from huggingface_hub import InferenceClient
from .base import BaseLLMProvider, Message, LLMResponse
from .tools import ToolSpec
import logging

logger = logging.getLogger(__name__)

class HuggingFaceProvider(BaseLLMProvider):
    """
    HuggingFace Inference API provider.

    Tool calling through the inference client is unreliable across served
    models, so this provider declares no support and the factory keeps it out
    of the fallback chain whenever tools are requested.
    """

    supports_tools = False

    
    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("HuggingFace token is required")
        self.client = InferenceClient(token=self.api_key, timeout=self.timeout)
    
    def generate_response(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        **kwargs
    ) -> LLMResponse:
        if tools:
            # Should be unreachable: the factory filters this provider out of
            # tool-calling requests. Refuse loudly rather than answer without
            # the tools and let the caller believe the data was consulted.
            return LLMResponse(
                content="", model=self.model, provider="huggingface",
                finish_reason="error",
                error="huggingface provider does not support tool calling",
            )
        try:
            formatted_messages = self.format_messages(messages, system_prompt)
            
            response = self.client.chat_completion(
                messages=formatted_messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                provider="huggingface",
                tokens_used=None,
                finish_reason=response.choices[0].finish_reason
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
            
            stream = self.client.chat_completion(
                messages=formatted_messages,
                model=self.model,
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
