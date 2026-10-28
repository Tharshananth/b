"""
LLM Provider Factory
Creates and manages LLM provider instances
"""
from typing import Optional, List, Dict
import logging
from config import get_config, get_provider_api_key
from .base import BaseLLMProvider, Message, LLMResponse
from .openai_provider import OpenAIProvider
from .claude_provider import ClaudeProvider
from .gemini_provider import GeminiProvider
from .huggingface_provider import HuggingFaceProvider

logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory for creating LLM providers"""
    
    PROVIDERS = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "huggingface": HuggingFaceProvider
    }
    
    def __init__(self):
        self.config = get_config()
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize all enabled providers"""
        for provider_name, provider_config in self.config.llm.providers.items():
            if not provider_config.enabled:
                logger.info(f"Provider {provider_name} is disabled")
                continue
            
            try:
                api_key = get_provider_api_key(provider_name)
                if not api_key:
                    logger.warning(f"No API key found for {provider_name}")
                    continue
                
                provider_class = self.PROVIDERS.get(provider_name)
                if not provider_class:
                    logger.warning(f"Unknown provider: {provider_name}")
                    continue
                
                # Create provider instance
                provider = provider_class(
                    api_key=api_key,
                    model=provider_config.model,
                    temperature=provider_config.temperature,
                    max_tokens=provider_config.max_tokens,
                    top_p=provider_config.top_p,
                    timeout=provider_config.timeout
                )
                
                self._providers[provider_name] = provider
                logger.info(f" Initialized {provider_name} provider")
                
            except Exception as e:
                logger.error(f" Failed to initialize {provider_name}: {e}")
    
    def get_provider(self, name: Optional[str] = None) -> Optional[BaseLLMProvider]:
        """
        Get a specific provider by name
        
        Args:
            name: Provider name (openai, claude, gemini, huggingface)
            
        Returns:
            Provider instance or None
        """
        if name is None:
            return self.get_default_provider()
        
        provider = self._providers.get(name)
        if not provider:
            logger.warning(f"Provider {name} not available")
        return provider
    
    def get_default_provider(self) -> Optional[BaseLLMProvider]:
        """Get the default provider"""
        default_name = self.config.llm.default_provider
        return self._providers.get(default_name)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self._providers.keys())
    
    def get_provider_info(self) -> List[Dict]:
        """Get information about all available providers"""
        return [
            {
                "name": name,
                **provider.get_info()
            }
            for name, provider in self._providers.items()
        ]
    
    def get_tool_capable_providers(self) -> List[str]:
        """Names of available providers that can accept tool definitions."""
        return [n for n, p in self._providers.items() if p.supports_tools]

    def generate_with_fallback(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        tools: Optional[List] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with automatic fallback to other providers

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            preferred_provider: Preferred provider name
            tools: Tool definitions. When given, only providers that support
                tools are eligible - falling back onto one that silently drops
                them would produce a confident answer with no data behind it.
            **kwargs: Additional parameters

        Returns:
            LLMResponse from the first successful provider
        """
        # Build provider order
        providers_to_try = []

        if preferred_provider and preferred_provider in self._providers:
            providers_to_try.append(preferred_provider)


        default = self.config.llm.default_provider
        if default not in providers_to_try and default in self._providers:
            providers_to_try.append(default)

        if self.config.llm.enable_fallback:
            for provider_name in self.config.llm.fallback_order:
                if provider_name not in providers_to_try and provider_name in self._providers:
                    providers_to_try.append(provider_name)

        if tools:
            eligible = [
                n for n in providers_to_try if self._providers[n].supports_tools
            ]
            dropped = [n for n in providers_to_try if n not in eligible]
            if dropped:
                logger.info(
                    "Tool request: excluding non-tool providers %s", dropped
                )
            providers_to_try = eligible

            if not providers_to_try:
                msg = "No tool-capable provider is available"
                logger.error(msg)
                return LLMResponse(
                    content="I can't look that up right now. Please try again "
                            "shortly.",
                    model="unknown", provider="none",
                    finish_reason="error", error=msg,
                )
            kwargs["tools"] = tools

        # Try providers in order
        last_error = None
        for provider_name in providers_to_try:
            try:
                provider = self._providers[provider_name]
                logger.info(f"Trying provider: {provider_name}")

                response = provider.generate_response(
                    messages=messages,
                    system_prompt=system_prompt,
                    **kwargs
                )
                
                if response.finish_reason != "error":
                    logger.info(f" Success with {provider_name}")
                    return response
                
                last_error = response.error
                logger.warning(f"Provider {provider_name} returned error: {response.error}")
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Provider {provider_name} failed: {e}")
                continue
        
        # All providers failed
        error_msg = f"All providers failed. Last error: {last_error}"
        logger.error(error_msg)
        
        return LLMResponse(
            content="I apologize, but I'm currently unable to process your request. Please try again later.",
            model="unknown",
            provider="none",
            finish_reason="error",
            error=error_msg
        )

# Global factory instance
_factory: Optional[LLMFactory] = None

def get_llm_factory() -> LLMFactory:
    """Get or create the global LLM factory instance"""
    global _factory
    if _factory is None:
        _factory = LLMFactory()
    return _factory

def get_llm_provider(name: Optional[str] = None) -> Optional[BaseLLMProvider]:
    """
    Convenience function to get a provider
    
    Args:
        name: Provider name or None for default
        
    Returns:
        Provider instance
    """
    factory = get_llm_factory()
    return factory.get_provider(name)