"""Vector store implementation using ChromaDB with Hybrid Search"""
import logging
from pathlib import Path
from typing import List, Optional
try:
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class Document:
        page_content: str = ""
        metadata: dict = field(default_factory=dict)
    Chroma = None
    RecursiveCharacterTextSplitter = None
    BM25Retriever = None
from config import get_config
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for document storage and retrieval with hybrid search"""
    
    def __init__(self):
        self.config = get_config()
        self.embeddings = get_embeddings()
        self.persist_directory = self.config.vector_db.persist_directory
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.vectordb = Chroma(
            collection_name=self.config.vector_db.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.vector_db.chunking.chunk_size,
            chunk_overlap=self.config.vector_db.chunking.chunk_overlap,
            separators=self.config.vector_db.chunking.separators
        )
        
        # Initialize BM25 index cache for keyword search
        self.bm25_retriever = None
        self._update_bm25_retriever()
        
        logger.info(f"Vector store initialized: {self.persist_directory}")
    
    def _update_bm25_retriever(self):
        """Build/refresh the in-memory BM25 index from current ChromaDB chunks"""
        try:
            db_data = self.vectordb.get()
            if db_data and db_data.get("documents"):
                docs = [
                    Document(page_content=text, metadata=meta or {})
                    for text, meta in zip(
                        db_data["documents"],
                        db_data.get("metadatas", [{}] * len(db_data["documents"]))
                    )
                ]
                self.bm25_retriever = BM25Retriever.from_documents(docs)
                logger.info(f"BM25 index updated with {len(docs)} chunks")
            else:
                self.bm25_retriever = None
                logger.info("BM25 index cleared — no documents in store")
        except Exception as e:
            logger.error(f"Failed to update BM25 index: {e}")
            self.bm25_retriever = None
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects
            
        Returns:
            Number of chunks added
        """
        if not documents:
            return 0
        
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vectordb.add_documents(chunks)
            
            # Refresh BM25 index to include new chunks
            self._update_bm25_retriever()
            
            logger.info(f"Added {len(chunks)} chunks from {len(documents)} documents")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0
    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for relevant documents (vector-only, kept for backward compatibility)
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        k = k or self.config.vector_db.search.k
        
        try:
            results = self.vectordb.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def search_hybrid(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Hybrid search combining dense vector similarity with sparse BM25 keyword matching.
        Results are deduplicated and filtered by relevance score.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum relevance score for vector results
            
        Returns:
            List of relevant documents with retrieval metadata
        """
        k = k or self.config.vector_db.search.k
        threshold = score_threshold or self.config.vector_db.search.score_threshold
        
        # ── 1. Dense Semantic Search (with score filtering) ──────────────
        vector_docs = []
        try:
            vector_results = self.vectordb.similarity_search_with_relevance_scores(
                query=query,
                k=k
            )
            for doc, score in vector_results:
                if score >= threshold:
                    doc.metadata["retrieval_score"] = round(float(score), 4)
                    doc.metadata["retrieval_method"] = "vector"
                    vector_docs.append(doc)
            
            logger.info(
                f"Vector search returned {len(vector_results)} results, "
                f"{len(vector_docs)} passed threshold ({threshold})"
            )
        except Exception as e:
            logger.error(f"Relevance-scored search failed, falling back to basic: {e}")
            # Fallback to standard similarity search without score filtering
            try:
                for doc in self.vectordb.similarity_search(query=query, k=k):
                    doc.metadata["retrieval_score"] = threshold
                    doc.metadata["retrieval_method"] = "vector"
                    vector_docs.append(doc)
            except Exception as e_inner:
                logger.error(f"Fallback vector search also failed: {e_inner}")
        
        # ── 2. Sparse Keyword Search (BM25) ──────────────────────────────
        bm25_docs = []
        if self.bm25_retriever:
            try:
                self.bm25_retriever.k = k
                bm25_results = self.bm25_retriever.invoke(query)
                for doc in bm25_results:
                    doc.metadata["retrieval_score"] = threshold
                    doc.metadata["retrieval_method"] = "bm25"
                    bm25_docs.append(doc)
                
                logger.info(f"BM25 search returned {len(bm25_docs)} results")
            except Exception as e:
                logger.error(f"BM25 retrieval failed: {e}")
        else:
            logger.debug("BM25 retriever not available, skipping keyword search")
        
        # ── 3. Ensemble Fusion & Deduplication ───────────────────────────
        # Vector results take priority; BM25 fills in keyword-matched gaps
        combined = {}
        for doc in vector_docs:
            combined[doc.page_content] = doc
        for doc in bm25_docs:
            if doc.page_content not in combined:
                combined[doc.page_content] = doc
        
        final_results = list(combined.values())[:k]
        logger.info(
            f"Hybrid search: {len(vector_docs)} vector + {len(bm25_docs)} BM25 "
            f"→ {len(final_results)} unique results (k={k})"
        )
        
        return final_results
    
    def search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[tuple[Document, float]]:
        """
        Search with relevance scores
        
        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum relevance score
            
        Returns:
            List of (document, score) tuples
        """
        k = k or self.config.vector_db.search.k
        score_threshold = score_threshold or self.config.vector_db.search.score_threshold
        
        try:
            results = self.vectordb.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by score threshold
            filtered = [
                (doc, score) for doc, score in results
                if score >= score_threshold
            ]
            
            return filtered
        except Exception as e:
            logger.error(f"Search with score error: {e}")
            return []
    
    def delete_collection(self) -> None:
        """Delete the entire collection and clear the BM25 index"""
        try:
            self.vectordb.delete_collection()
            self.bm25_retriever = None
            logger.info("Collection deleted and BM25 index cleared")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_document_count(self) -> int:
        """Get number of documents in the collection (using safe public API)"""
        try:
            db_data = self.vectordb.get()
            return len(db_data.get("ids", []))
        except:
            return 0