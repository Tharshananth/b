# routers/config.py
"""Configuration management endpoints"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
import logging
from llm.factory import get_llm_factory
from config import get_config
from auth.dependencies import require_admin

logger = logging.getLogger(__name__)

# Removed on purpose (V2 item 2): POST /provider/{name} (model is set in config.yaml on the server, never over HTTP) and GET /system-prompt (exposed the system prompt).
router = APIRouter(
    prefix="/api/config",
    tags=["Configuration"],
    dependencies=[Depends(require_admin)],
)


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
        logger.error(f"Config error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


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
        logger.error(f"Provider list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve provider list")
