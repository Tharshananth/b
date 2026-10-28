
# routers/health.py
"""Health check endpoints"""
from fastapi import APIRouter
from typing import Dict
import logging
from datetime import datetime
from llm.factory import get_llm_factory
from vector_db.retriever import get_retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/health", tags=["Health"])

@router.get("/")
async def health_check() -> Dict:
    """
    Comprehensive health check
    
    Returns:
        Health status of all services
    """
    try:
        # Check LLM providers
        factory = get_llm_factory()
        available_providers = factory.get_available_providers()
        
        # Check vector database
        try:
            retriever = get_retriever()
            doc_count = retriever.get_vector_store().get_document_count()
            vector_db_status = "healthy"
        except Exception as e:
            logger.error(f"Vector DB health check failed: {e}")
            doc_count = 0
            vector_db_status = "unhealthy"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "healthy",
                "vector_db": vector_db_status,
                "llm_providers": {
                    "available": available_providers,
                    "count": len(available_providers)
                }
            },
            "metrics": {
                "documents_indexed": doc_count
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/ready")
async def readiness_check() -> Dict:
    """
    Readiness check for k8s/deployment
    
    Returns:
        Ready status
    """
    try:
        factory = get_llm_factory()
        providers = factory.get_available_providers()
        
        if not providers:
            return {
                "ready": False,
                "reason": "No LLM providers available"
            }
        
        return {
            "ready": True,
            "providers": providers
        }
    except Exception as e:
        return {
            "ready": False,
            "reason": str(e)
        }

@router.get("/live")
async def liveness_check() -> Dict:
    """
    Liveness check for k8s/deployment
    
    Returns:
        Live status
    """
    return {"alive": True}

