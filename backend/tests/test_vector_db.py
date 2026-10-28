"""Tests for vector database"""
import pytest
from pathlib import Path
from langchain.schema import Document
from vector_db.store import VectorStore
from vector_db.retriever import DocumentRetriever
from vector_db.embeddings import get_embeddings

def test_embeddings_initialization():
    """Test embeddings initialization"""
    embeddings = get_embeddings()
    assert embeddings is not None

def test_embed_query():
    """Test embedding a query"""
    embeddings = get_embeddings()
    result = embeddings.embed_query("test query")
    assert isinstance(result, list)
    assert len(result) > 0

def test_embed_documents():
    """Test embedding multiple documents"""
    embeddings = get_embeddings()
    texts = ["document 1", "document 2", "document 3"]
    results = embeddings.embed_documents(texts)
    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)

def test_vector_store_initialization():
    """Test vector store initialization"""
    store = VectorStore()
    assert store is not None
    assert store.vectordb is not None

def test_add_documents():
    """Test adding documents to vector store"""
    store = VectorStore()
    
    docs = [
        Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        Document(page_content="Test document 2", metadata={"source": "test2.txt"})
    ]
    
    count = store.add_documents(docs)
    assert count > 0

def test_search_documents():
    """Test searching documents"""
    store = VectorStore()
    
    # Add test documents
    docs = [
        Document(page_content="Python is a programming language", metadata={"source": "test.txt"}),
        Document(page_content="JavaScript is used for web development", metadata={"source": "test.txt"})
    ]
    store.add_documents(docs)
    
    # Search
    results = store.search("programming", k=1)
    assert len(results) > 0
    assert "Python" in results[0].page_content or "programming" in results[0].page_content.lower()

def test_document_retriever():
    """Test document retriever"""
    retriever = DocumentRetriever()
    
    context_data = retriever.retrieve_context("What is PingUs?")
    assert "context" in context_data
    assert "sources" in context_data
    assert isinstance(context_data["sources"], list)
