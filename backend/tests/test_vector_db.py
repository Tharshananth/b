"""Tests for vector database

NOTE: This file imports Document from the local vector_db package (which
provides a pure-Python fallback dataclass when langchain is not installed).
All heavy operations (Chroma, embeddings) are skipped when their deps are
absent.
"""
import sys
import types
import importlib.util
from pathlib import Path
import pytest

# ── Ensure the local backend directory is on sys.path ──────────────────────
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# ── Isolate vector_db from any stubs placed by test_medical_guardrail.py ───
# If a bare (stubbed) vector_db is in sys.modules we reload the real one.
_vdb_key = "vector_db"
_vdb_path = _backend_dir / "vector_db" / "__init__.py"

def _real_vdb_loaded():
    """Return True when the real vector_db module exposes Document."""
    m = sys.modules.get(_vdb_key)
    return m is not None and hasattr(m, "Document")

if not _real_vdb_loaded():
    # Remove any stub so importlib loads the real package
    for key in list(sys.modules.keys()):
        if key == "vector_db" or key.startswith("vector_db."):
            del sys.modules[key]

# Import Document (the local package provides a pure-Python fallback)
from vector_db import Document
from vector_db.store import VectorStore
from vector_db.retriever import DocumentRetriever
from vector_db.embeddings import get_embeddings

# Detect whether real embedding backends are available
try:
    from langchain_openai import OpenAIEmbeddings as _OAE  # noqa: F401
    _HAS_EMBEDDINGS = True
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings as _HFE  # noqa: F401
        _HAS_EMBEDDINGS = True
    except ImportError:
        _HAS_EMBEDDINGS = False

try:
    from langchain_chroma import Chroma as _Chroma  # noqa: F401
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

_SKIP_EMBED = pytest.mark.skipif(not _HAS_EMBEDDINGS, reason="No embedding backend installed")
_SKIP_STORE = pytest.mark.skipif(not (_HAS_EMBEDDINGS and _HAS_CHROMA), reason="langchain_chroma or embedding backend not installed")


# ── Tests ───────────────────────────────────────────────────────────────────

def test_document_dataclass():
    """Document dataclass is importable and usable without langchain."""
    doc = Document(page_content="hello", metadata={"source": "test.txt"})
    assert doc.page_content == "hello"
    assert doc.metadata["source"] == "test.txt"


@_SKIP_EMBED
def test_embeddings_initialization():
    """Test embeddings initialization"""
    embeddings = get_embeddings()
    assert embeddings is not None


@_SKIP_EMBED
def test_embed_query():
    """Test embedding a query"""
    embeddings = get_embeddings()
    result = embeddings.embed_query("test query")
    assert isinstance(result, list)
    assert len(result) > 0


@_SKIP_EMBED
def test_embed_documents():
    """Test embedding multiple documents"""
    embeddings = get_embeddings()
    texts = ["document 1", "document 2", "document 3"]
    results = embeddings.embed_documents(texts)
    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)


@_SKIP_STORE
def test_vector_store_initialization():
    """Test vector store initialization"""
    store = VectorStore()
    assert store is not None
    assert store.vectordb is not None


@_SKIP_STORE
def test_add_documents():
    """Test adding documents to vector store"""
    store = VectorStore()

    docs = [
        Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        Document(page_content="Test document 2", metadata={"source": "test2.txt"})
    ]

    count = store.add_documents(docs)
    assert count > 0


@_SKIP_STORE
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


@_SKIP_STORE
def test_document_retriever():
    """Test document retriever"""
    retriever = DocumentRetriever()

    context_data = retriever.retrieve_context("What is PingUs?")
    assert "context" in context_data
    assert "sources" in context_data
    assert isinstance(context_data["sources"], list)
