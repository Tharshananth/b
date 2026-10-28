

# routers/config.py
"""Configuration management endpoints"""
from fastapi import APIRouter, HTTPException
from typing import Dict, List
import logging
from llm.factory import get_llm_factory
from config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/config", tags=["Configuration"])

@router.get("/")
async def get_configuration() -> Dict:
    """
    Get current configuration
    
    Returns:
        Current configuration details
    """
    try:
        config = get_config()
        factory = get_llm_factory()
        
        return {
            "app": {
                "name": config.app.name,
                "version": config.app.version,
                "environment": config.app.environment
            },
            "current_provider": config.llm.default_provider,
            "available_providers": factory.get_provider_info(),
            "embedding_provider": config.embeddings.provider,
            "vector_db": {
                "type": config.vector_db.type,
                "collection": config.vector_db.collection_name
            }
        }
    except Exception as e:
        logger.error(f"Config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/providers")
async def list_providers() -> List[Dict]:
    """
    List all available LLM providers
    
    Returns:
        List of provider details
    """
    try:
        factory = get_llm_factory()
        return factory.get_provider_info()
    except Exception as e:
        logger.error(f"Provider list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/provider/{provider_name}")
async def switch_provider(provider_name: str) -> Dict:
    """
    Switch to a different LLM provider
    
    Args:
        provider_name: Name of provider to switch to
        
    Returns:
        Switch status
    """
    try:
        factory = get_llm_factory()
        available = factory.get_available_providers()
        
        if provider_name not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Provider {provider_name} not available. Available: {available}"
            )
        
        # Update default provider in config (runtime only)
        config = get_config()
        config.llm.default_provider = provider_name
        
        logger.info(f"Switched to provider: {provider_name}")
        
        return {
            "success": True,
            "provider": provider_name,
            "message": f"Switched to {provider_name} provider"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Provider switch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system-prompt")
async def get_system_prompt() -> Dict:
    """
    Get the current system prompt
    
    Returns:
        System prompt text
    """
    try:
        config = get_config()
        return {
            "system_prompt": config.system_prompt
        }
    except Exception as e:
        logger.error(f"System prompt error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        