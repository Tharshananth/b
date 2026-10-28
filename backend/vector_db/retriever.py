
# vector_db/retriever.py
"""Document retriever for RAG with hybrid search support"""
import logging
from typing import List, Dict, Optional
from .store import VectorStore

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Retrieves relevant documents for RAG using hybrid search"""
    
    def __init__(self):
        self.vector_store = VectorStore()
    
    def retrieve_context(
        self,
        query: str,
        k: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Retrieve relevant context for a query using hybrid search
        (dense vector + sparse BM25 keyword matching with score filtering)
        
        Args:
            query: User query
            k: Number of documents to retrieve (defaults to config value)
            
        Returns:
            Dictionary with context and sources
        """
        try:
            config = self.vector_store.config
            k = k or config.vector_db.search.k
            threshold = config.vector_db.search.score_threshold
            
            # Execute hybrid search (vector + BM25)
            results = self.vector_store.search_hybrid(
                query=query,
                k=k,
                score_threshold=threshold
            )
            
            if not results:
                return {
                    "context": "No relevant information found.",
                    "sources": []
                }
            
            # Build context from results
            context_parts = []
            sources = []
            
            for i, doc in enumerate(results, 1):
                context_parts.append(f"[{i}] {doc.page_content}")
                
                # Log retrieval metadata for observability
                method = doc.metadata.get("retrieval_method", "unknown")
                score = doc.metadata.get("retrieval_score", 0.0)
                source_name = doc.metadata.get("source", "Unknown")
                logger.info(
                    f"Retrieved chunk [{i}] from '{source_name}' "
                    f"via {method} (score: {score:.3f})"
                )
                
                # Extract source info
                source = {
                    "title": source_name,
                    "url": doc.metadata.get("url", "#"),
                    "content": doc.page_content[:200] + "..."
                }
                sources.append(source)
            
            context = "\n\n".join(context_parts)
            
            return {
                "context": context,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {
                "context": "Error retrieving information.",
                "sources": []
            }
    
    def get_vector_store(self) -> VectorStore:
        """Get the underlying vector store"""
        return self.vector_store


# Constructing a DocumentRetriever opens Chroma and rebuilds the whole BM25
# index from a full dump of the collection. Doing that per request made every
# chat message pay for a reindex of the entire knowledge base, so the instance
# is built once and shared.
_retriever: Optional[DocumentRetriever] = None


def get_retriever() -> DocumentRetriever:
    """Get the shared retriever, building it on first use."""
    global _retriever
    if _retriever is None:
        logger.info("Building shared DocumentRetriever")
        _retriever = DocumentRetriever()
    return _retriever


def reset_retriever() -> None:
    """
    Drop the cached retriever.

    Required after delete_collection(): the cached instance would otherwise keep
    a handle to a collection that no longer exists.
    """
    global _retriever
    _retriever = None
    logger.info("Shared DocumentRetriever reset")