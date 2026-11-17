"""Tests for vector similarity score fixes (Step 1-3)."""
import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend is on path
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from vector_db.store import distance_to_cosine

# ── Test 1: distance_to_cosine ─────────────────────────────────────────────

class TestDistanceToCosine:
    def test_l2_zero(self):
        assert distance_to_cosine(0.0, "l2") == pytest.approx(1.0)

    def test_l2_one(self):
        assert distance_to_cosine(1.0, "l2") == pytest.approx(0.5)

    def test_l2_two(self):
        assert distance_to_cosine(2.0, "l2") == pytest.approx(0.0)

    def test_l2_clamped(self):
        assert distance_to_cosine(4.0, "l2") == 0.0  # clamped, not negative

    def test_cosine_space(self):
        assert distance_to_cosine(0.3, "cosine") == pytest.approx(0.7)

    def test_ip_space(self):
        assert distance_to_cosine(0.3, "ip") == pytest.approx(0.7)

    def test_unknown_space_uses_l2_rule(self):
        result = distance_to_cosine(1.0, "unknown_xyz")
        assert result == pytest.approx(0.5)  # l2 formula: 1 - 1/2 = 0.5

    def test_unknown_space_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vector_db.store"):
            distance_to_cosine(0.0, "new_unknown_space_test")
        assert "Unknown Chroma distance space" in caplog.text


# ── Test 2: hybrid search with fake vectordb ───────────────────────────────

class TestHybridSearchConversion:
    """distances [0.9, 1.2, 1.5] → cosines [0.55, 0.40, 0.25]; threshold 0.40 keeps first two."""

    def _make_store_with_mock_vdb(self, distances_and_docs):
        """Create a VectorStore with mock internals, bypassing __init__."""
        from vector_db.store import VectorStore
        store = object.__new__(VectorStore)
        
        cfg = MagicMock()
        cfg.vector_db.search.k = 5
        cfg.vector_db.search.score_threshold = 0.40
        cfg.vector_db.chunking.chunk_size = 500
        cfg.vector_db.chunking.chunk_overlap = 50
        cfg.vector_db.chunking.separators = ["\n\n"]
        cfg.vector_db.persist_directory = "./chroma_db"
        cfg.vector_db.collection_name = "test"
        store.config = cfg
        store._distance_space = "l2"
        store.bm25_retriever = None
        
        mock_vdb = MagicMock()
        mock_vdb.similarity_search_with_score.return_value = distances_and_docs
        store.vectordb = mock_vdb
        return store

    def test_hybrid_keeps_above_threshold(self):
        from langchain_core.documents import Document
        docs_and_distances = [
            (Document(page_content="A", metadata={}), 0.9),   # cosine=0.55
            (Document(page_content="B", metadata={}), 1.2),   # cosine=0.40
            (Document(page_content="C", metadata={}), 1.5),   # cosine=0.25
        ]
        store = self._make_store_with_mock_vdb(docs_and_distances)
        result = store.search_hybrid("test query", k=5, score_threshold=0.40)
        
        contents = [d.page_content for d in result]
        assert "A" in contents
        assert "B" in contents
        assert "C" not in contents

    def test_hybrid_retrieval_score_set(self):
        from langchain_core.documents import Document
        docs_and_distances = [
            (Document(page_content="A", metadata={}), 0.9),
            (Document(page_content="B", metadata={}), 1.2),
        ]
        store = self._make_store_with_mock_vdb(docs_and_distances)
        result = store.search_hybrid("test query", k=5, score_threshold=0.40)
        
        for doc in result:
            assert "retrieval_score" in doc.metadata
            assert 0.0 <= doc.metadata["retrieval_score"] <= 1.0
            assert doc.metadata["retrieval_method"] == "vector"

    def test_hybrid_order_most_similar_first(self):
        from langchain_core.documents import Document
        # Input order from similarity_search_with_score is most similar first:
        # distance 0.9 (cosine 0.55) then distance 1.2 (cosine 0.40)
        docs_and_distances = [
            (Document(page_content="Doc1", metadata={}), 0.9),
            (Document(page_content="Doc2", metadata={}), 1.2),
        ]
        store = self._make_store_with_mock_vdb(docs_and_distances)
        result = store.search_hybrid("test query", k=5, score_threshold=0.35)
        assert len(result) == 2
        assert result[0].page_content == "Doc1"
        assert result[1].page_content == "Doc2"
        assert result[0].metadata["retrieval_score"] > result[1].metadata["retrieval_score"]


# ── Test 3: search_with_score keeps MORE similar (regression for inversion) ──

class TestSearchWithScoreNotInverted:
    def test_keeps_more_similar(self):
        from langchain_core.documents import Document
        from vector_db.store import VectorStore
        
        store = object.__new__(VectorStore)
        cfg = MagicMock()
        cfg.vector_db.search.k = 5
        cfg.vector_db.search.score_threshold = 0.40
        store.config = cfg
        store._distance_space = "l2"
        
        mock_vdb = MagicMock()
        # distance=0.5 → cosine=0.75 (similar), distance=1.8 → cosine=0.1 (dissimilar)
        mock_vdb.similarity_search_with_score.return_value = [
            (Document(page_content="Similar doc", metadata={}), 0.5),
            (Document(page_content="Dissimilar doc", metadata={}), 1.8),
        ]
        store.vectordb = mock_vdb
        
        result = store.search_with_score("test", k=5, score_threshold=0.40)
        contents = [d.page_content for d, _ in result]
        
        assert "Similar doc" in contents
        assert "Dissimilar doc" not in contents

    def test_sorted_highest_first(self):
        from langchain_core.documents import Document
        from vector_db.store import VectorStore
        
        store = object.__new__(VectorStore)
        cfg = MagicMock()
        cfg.vector_db.search.k = 5
        cfg.vector_db.search.score_threshold = 0.0
        store.config = cfg
        store._distance_space = "l2"
        
        mock_vdb = MagicMock()
        mock_vdb.similarity_search_with_score.return_value = [
            (Document(page_content="Less", metadata={}), 1.0),   # cosine=0.5
            (Document(page_content="More", metadata={}), 0.2),   # cosine=0.9
        ]
        store.vectordb = mock_vdb
        
        result = store.search_with_score("test", k=5, score_threshold=0.0)
        assert result[0][0].page_content == "More"  # highest cosine first (0.9)
        assert result[1][0].page_content == "Less"  # second (0.5)
        assert result[0][1] == pytest.approx(0.9)
        assert result[1][1] == pytest.approx(0.5)


# ── Test 4: HuggingFaceEmbeddings gets normalize_embeddings=True ───────────

class TestNormalizeEmbeddings:
    def test_huggingface_normalize_flag(self):
        """HuggingFaceEmbeddings must be constructed with normalize_embeddings=True."""
        captured = {}
        
        class FakeHF:
            def __init__(self, **kwargs):
                captured.update(kwargs)
        
        import vector_db.embeddings as emb_mod
        orig = emb_mod.HuggingFaceEmbeddings
        orig_cache = emb_mod._embeddings
        emb_mod._embeddings = None
        emb_mod.HuggingFaceEmbeddings = FakeHF
        
        try:
            emb_mod.get_embeddings()
        except Exception:
            pass
        finally:
            emb_mod.HuggingFaceEmbeddings = orig
            emb_mod._embeddings = orig_cache
        
        assert "encode_kwargs" in captured
        assert captured["encode_kwargs"].get("normalize_embeddings") is True


# ── Test 5: No "Relevance scores must be between 0 and 1" warning ──────────

class TestNoRelevanceScoreWarning:
    def test_no_relevance_score_warning(self, recwarn):
        """search_hybrid must not emit the LangChain relevance-score warning."""
        from langchain_core.documents import Document
        from vector_db.store import VectorStore
        
        store = object.__new__(VectorStore)
        cfg = MagicMock()
        cfg.vector_db.search.k = 3
        cfg.vector_db.search.score_threshold = 0.40
        store.config = cfg
        store._distance_space = "l2"
        store.bm25_retriever = None
        
        mock_vdb = MagicMock()
        mock_vdb.similarity_search_with_score.return_value = [
            (Document(page_content="X", metadata={}), 0.8),  # cosine=0.6
        ]
        store.vectordb = mock_vdb
        
        store.search_hybrid("test", k=3, score_threshold=0.40)
        
        relevance_warnings = [
            w for w in recwarn.list
            if "Relevance scores must be between" in str(w.message)
        ]
        assert len(relevance_warnings) == 0
