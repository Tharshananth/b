# Implementation Plan — RAG Retrieval Optimization

This plan details the side-by-side optimization of the document retrieval pipeline in the BSAI Chatbot.

---

## Proposed Changes

### 1. Vector Store Optimization

#### [MODIFY] [store.py](file:///c:/Users/tharshananth%20N/brainsai/bsai_chatbot/backend/vector_db/store.py)
*   **Current State:** Performs simple vector-only search and uses private API `_collection.count()`. No keyword matching or score thresholding.
*   **Optimized State:** 
    *   Builds and caches an in-memory BM25 index on startup and dynamically updates it when documents are added/cleared.
    *   Executes hybrid search (combining semantic scores with BM25 keyword matches).
    *   Replaces private count call with standard `self.vectordb.get()["ids"]` lookup.

##### Comparison (Current vs. Optimized)

````carousel
```python
# CURRENT: store.py (Excerpt)

class VectorStore:
    def __init__(self):
        ...
        self.vectordb = Chroma(
            collection_name=self.config.vector_db.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.text_splitter = RecursiveCharacterTextSplitter(...)

    def add_documents(self, documents: List[Document]) -> int:
        if not documents: return 0
        try:
            chunks = self.text_splitter.split_documents(documents)
            self.vectordb.add_documents(chunks)
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding: {e}")
            return 0

    def search(self, query: str, k: Optional[int] = None, filter: Optional[dict] = None):
        k = k or self.config.vector_db.search.k
        try:
            return self.vectordb.similarity_search(query, k=k, filter=filter)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_document_count(self) -> int:
        try:
            return self.vectordb._collection.count()
        except:
            return 0
```
<!-- slide -->
```python
# OPTIMIZED: store.py (Excerpt)

from langchain_community.retrievers import BM25Retriever

class VectorStore:
    def __init__(self):
        ...
        self.vectordb = Chroma(
            collection_name=self.config.vector_db.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.text_splitter = RecursiveCharacterTextSplitter(...)
        
        # Initialize BM25 index cache
        self.bm25_retriever = None
        self._update_bm25_retriever()

    def _update_bm25_retriever(self):
        """Update BM25 index from current database chunks"""
        try:
            db_data = self.vectordb.get()
            if db_data and db_data.get("documents"):
                docs = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(db_data["documents"], db_data["metadatas"])
                ]
                self.bm25_retriever = BM25Retriever.from_documents(docs)
                logger.info(f"BM25 index updated with {len(docs)} chunks")
            else:
                self.bm25_retriever = None
        except Exception as e:
            logger.error(f"Failed to update BM25 index: {e}")
            self.bm25_retriever = None

    def add_documents(self, documents: List[Document]) -> int:
        if not documents: return 0
        try:
            chunks = self.text_splitter.split_documents(documents)
            self.vectordb.add_documents(chunks)
            self._update_bm25_retriever() # Refresh BM25
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding: {e}")
            return 0

    def delete_collection(self) -> None:
        try:
            self.vectordb.delete_collection()
            self.bm25_retriever = None
            logger.info("Collection deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def search_hybrid(
        self, 
        query: str, 
        k: Optional[int] = None, 
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Hybrid BM25 + Vector Search with relevance filtering"""
        k = k or self.config.vector_db.search.k
        threshold = score_threshold or self.config.vector_db.search.score_threshold
        
        # 1. Dense Semantic Search
        vector_docs = []
        try:
            vector_results = self.vectordb.similarity_search_with_relevance_scores(query, k=k)
            for doc, score in vector_results:
                if score >= threshold:
                    doc.metadata["retrieval_score"] = float(score)
                    doc.metadata["retrieval_method"] = "vector"
                    vector_docs.append(doc)
        except Exception as e:
            logger.error(f"Relevance scoring search failed, falling back: {e}")
            # Fallback to standard similarity search
            try:
                for doc in self.vectordb.similarity_search(query, k=k):
                    doc.metadata["retrieval_score"] = threshold
                    doc.metadata["retrieval_method"] = "vector"
                    vector_docs.append(doc)
            except Exception as e_inner:
                logger.error(f"Fallback search failed: {e_inner}")

        # 2. Sparse Keyword Search (BM25)
        bm25_docs = []
        if self.bm25_retriever:
            try:
                self.bm25_retriever.k = k
                for doc in self.bm25_retriever.invoke(query):
                    doc.metadata["retrieval_score"] = threshold
                    doc.metadata["retrieval_method"] = "bm25"
                    bm25_docs.append(doc)
            except Exception as e:
                logger.error(f"BM25 retrieval failed: {e}")

        # 3. Ensemble Fusion & Deduplication
        combined = {}
        for doc in vector_docs:
            combined[doc.page_content] = doc
        for doc in bm25_docs:
            if doc.page_content not in combined:
                combined[doc.page_content] = doc
                
        return list(combined.values())[:k]

    def get_document_count(self) -> int:
        """Safe public API count check"""
        try:
            db_data = self.vectordb.get()
            return len(db_data.get("ids", []))
        except:
            return 0
```
````

---

### 2. Context Retriever Refactoring

#### [MODIFY] [retriever.py](file:///c:/Users/tharshananth%20N/brainsai/bsai_chatbot/backend/vector_db/retriever.py)
*   **Current State:** Calls simple `search` (vector only) and doesn't pass relevance threshold constraints.
*   **Optimized State:** Calls `search_hybrid` and injects retrieval metadata (score and method) into logging.

##### Comparison (Current vs. Optimized)

````carousel
```python
# CURRENT: retriever.py (Excerpt)

class DocumentRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
    
    def retrieve_context(self, query: str, k: int = 4) -> Dict[str, any]:
        try:
            # Simple search call
            results = self.vector_store.search(query, k=k)
            if not results:
                return {"context": "No relevant information found.", "sources": []}
            
            context_parts = []
            sources = []
            for i, doc in enumerate(results, 1):
                context_parts.append(f"[{i}] {doc.page_content}")
                sources.append({
                    "title": doc.metadata.get("source", "Unknown"),
                    "url": doc.metadata.get("url", "#"),
                    "content": doc.page_content[:200] + "..."
                })
            return {"context": "\n\n".join(context_parts), "sources": sources}
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {"context": "Error retrieving information.", "sources": []}
```
<!-- slide -->
```python
# OPTIMIZED: retriever.py (Excerpt)

class DocumentRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
    
    def retrieve_context(self, query: str, k: Optional[int] = None) -> Dict[str, any]:
        try:
            config = self.vector_store.config
            k = k or config.vector_db.search.k
            threshold = config.vector_db.search.score_threshold
            
            # Hybrid search execution
            results = self.vector_store.search_hybrid(query, k=k, score_threshold=threshold)
            if not results:
                return {"context": "No relevant information found.", "sources": []}
            
            context_parts = []
            sources = []
            for i, doc in enumerate(results, 1):
                context_parts.append(f"[{i}] {doc.page_content}")
                
                # Capture retrieval stats in sources metadata
                method = doc.metadata.get("retrieval_method", "unknown")
                score = doc.metadata.get("retrieval_score", 0.0)
                logger.info(f"Retrieved chunk [{i}] via {method} (score: {score:.3f})")
                
                sources.append({
                    "title": doc.metadata.get("source", "Unknown"),
                    "url": doc.metadata.get("url", "#"),
                    "content": doc.page_content[:200] + "..."
                })
            return {"context": "\n\n".join(context_parts), "sources": sources}
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {"context": "Error retrieving information.", "sources": []}
```
````

---

## Verification Plan

### Automated Verification
*   We will start the FastAPI server and perform direct searches via curl/requests to check if RAG queries return context augmented with BM25 matches.

### Manual Verification
*   Verify exact matches like specific model versions are correctly found by the hybrid retriever compared to vector search alone.
