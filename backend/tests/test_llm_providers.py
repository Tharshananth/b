"""Tests for LLM providers"""
import pytest
from llm.base import Message, LLMResponse
from llm.factory import LLMFactory, get_llm_factory

def test_llm_factory_initialization():
    """Test LLM factory initialization"""
    factory = LLMFactory()
    assert factory is not None
    assert factory.config is not None

def test_get_available_providers():
    """Test getting available providers"""
    factory = get_llm_factory()
    providers = factory.get_available_providers()
    assert isinstance(providers, list)

def test_create_provider():
    """Test creating a provider"""
    factory = get_llm_factory()
    available = factory.get_available_providers()
    
    if available:
        provider = factory.get_provider(available[0])
        assert provider is not None

@pytest.mark.asyncio
async def test_generate_response():
    """Test generating a response"""
    factory = get_llm_factory()
    available = factory.get_available_providers()
    
    if not available:
        pytest.skip("No LLM providers available")
    
    messages = [Message(role="user", content="Hello, how are you?")]
    response = factory.generate_with_fallback(messages)
    
    assert isinstance(response, LLMResponse)
    assert response.content
    assert response.provider

@pytest.mark.asyncio
async def test_streaming_response():
    """Test streaming response"""
    factory = get_llm_factory()
    provider = factory.get_default_provider()
    
    if not provider:
        pytest.skip("No default provider available")
    
    messages = [Message(role="user", content="Count to 5")]
    
    chunks = []
    async for chunk in provider.stream_response(messages):
        chunks.append(chunk)
    
    assert len(chunks) > 0

def test_provider_fallback():
    """Test automatic fallback between providers"""
    factory = get_llm_factory()
    messages = [Message(role="user", content="Test message")]
    
    # This should try all providers and return a response
    response = factory.generate_with_fallback(messages)
    assert isinstance(response, LLMResponse)
