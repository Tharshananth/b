
# vector_db/embeddings.py
"""Embedding provider management"""
import logging
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config import get_config, get_embedding_api_key

logger = logging.getLogger(__name__)

_embeddings = None

def get_embeddings():
    """Get or create embeddings instance"""
    global _embeddings
    
    if _embeddings is not None:
        return _embeddings
    
    config = get_config()
    provider = config.embeddings.provider
    provider_config = config.embeddings.providers[provider]
    
    try:
        if provider == "openai":
            api_key = get_embedding_api_key()
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            _embeddings = OpenAIEmbeddings(
                model=provider_config.model,
                openai_api_key=api_key
            )
            logger.info(f" Using OpenAI embeddings: {provider_config.model}")
            
        elif provider == "huggingface":
            _embeddings = HuggingFaceEmbeddings(
                model_name=provider_config.model
            )
            logger.info(f" Using HuggingFace embeddings: {provider_config.model}")
        
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
        
        return _embeddings
        
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        # Fallback to HuggingFace
        logger.info("Falling back to HuggingFace embeddings")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return _embeddings

