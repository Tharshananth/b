"""
Pytest suite for prompts loading, citations metadata, and verification (V2 item 6).
No external network calls, no live vector DB / LLM SDK initialization.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# 0. Resolve paths
# ─────────────────────────────────────────────────────────────────────────────

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load llm.base without triggering llm/__init__.py
# ─────────────────────────────────────────────────────────────────────────────

llm_dir = str(backend_dir / "llm")
if "llm" not in sys.modules:
    bare_llm = types.ModuleType("llm")
    bare_llm.__path__ = [llm_dir]
    sys.modules["llm"] = bare_llm
else:
    bare_llm = sys.modules["llm"]

if "llm.base" in sys.modules:
    llm_base_mod = sys.modules["llm.base"]
else:
    spec = importlib.util.spec_from_file_location("llm.base", backend_dir / "llm" / "base.py")
    llm_base_mod = importlib.util.module_from_spec(spec)
    sys.modules["llm.base"] = llm_base_mod
    bare_llm.base = llm_base_mod
    spec.loader.exec_module(llm_base_mod)

from llm.base import Message, ToolCall

# ─────────────────────────────────────────────────────────────────────────────
# 2. Stub heavy dependencies before any router/services import
# ─────────────────────────────────────────────────────────────────────────────

_factory_mod = types.ModuleType("llm.factory")
_factory_mod.LLMFactory = object
_factory_mod.get_llm_factory = lambda: types.SimpleNamespace(
    get_available_providers=lambda: ["gemini"],
    get_provider_info=lambda: [],
)
sys.modules.setdefault("llm.factory", _factory_mod)

# Stub vector_db package and load real retriever.py
vdb_dir = str(backend_dir / "vector_db")
bare_vdb = types.ModuleType("vector_db")
bare_vdb.__path__ = [vdb_dir]
sys.modules.setdefault("vector_db", bare_vdb)

mock_store_mod = types.ModuleType("vector_db.store")
mock_store_mod.VectorStore = MagicMock
sys.modules["vector_db.store"] = mock_store_mod

spec_ret = importlib.util.spec_from_file_location("vector_db.retriever", backend_dir / "vector_db" / "retriever.py")
ret_mod = importlib.util.module_from_spec(spec_ret)
sys.modules["vector_db.retriever"] = ret_mod
bare_vdb.retriever = ret_mod
spec_ret.loader.exec_module(ret_mod)

# Stub agents package and load real prompt.py
agents_dir = str(backend_dir / "agents")
bare_agents = types.ModuleType("agents")
bare_agents.__path__ = [agents_dir]
sys.modules.setdefault("agents", bare_agents)

spec_prompt = importlib.util.spec_from_file_location("agents.prompt", backend_dir / "agents" / "prompt.py")
prompt_mod = importlib.util.module_from_spec(spec_prompt)
sys.modules["agents.prompt"] = prompt_mod
bare_agents.prompt = prompt_mod
spec_prompt.loader.exec_module(prompt_mod)

services_dir = str(backend_dir / "services")
_svc_mod = types.ModuleType("services")
_svc_mod.__path__ = [services_dir]
_orch_mock = MagicMock()
_sug_mock = MagicMock()
_sug_mock.build.return_value = []
_svc_mod.orchestrator = _orch_mock
_svc_mod.suggestions = _sug_mock
sys.modules.setdefault("services", _svc_mod)

if "services.job_status" not in sys.modules:
    spec_job = importlib.util.spec_from_file_location("services.job_status", backend_dir / "services" / "job_status.py")
    job_mod = importlib.util.module_from_spec(spec_job)
    spec_job.loader.exec_module(job_mod)
    sys.modules["services.job_status"] = job_mod
    _svc_mod.job_status = job_mod

routers_dir = str(backend_dir / "routers")
if "routers" not in sys.modules:
    bare_routers = types.ModuleType("routers")
    bare_routers.__path__ = [routers_dir]
    sys.modules["routers"] = bare_routers
else:
    bare_routers = sys.modules["routers"]

from routers.chat import router as chat_router
import routers.chat as _chat_module
bare_routers.chat = _chat_module

for _r_name in ["documents", "health", "config", "feedback", "auth"]:
    try:
        _m = importlib.import_module(f"routers.{_r_name}")
        setattr(bare_routers, _r_name, _m)
        setattr(bare_routers, f"{_r_name}_router", getattr(_m, "router", None))
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 3. Temp working directory with config.yaml + fresh SQLite DB
# ─────────────────────────────────────────────────────────────────────────────

_tmp = Path(tempfile.mkdtemp(prefix="bsai_citations_test_"))
_orig_cwd = os.getcwd()
os.chdir(_tmp)
shutil.copy(backend_dir / "config.yaml", _tmp / "config.yaml")
(_tmp / "data" / "database").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("JWT_SECRET", "test-secret-at-least-32-chars-long-abcdef123456")
os.environ.pop("GOOGLE_API_KEY", None)

from database.connection import init_db, SessionLocal
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Minimal FastAPI test app
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.chat import router as chat_router
from routers.health import router as health_router

_app = FastAPI()
_app.include_router(chat_router)
_app.include_router(health_router)


def _make_client():
    return TestClient(_app, raise_server_exceptions=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStep1Equivalence:
    """Test 1: get_react_system_prompt() built from prompt files matches expected structure."""

    def test_step1_equivalence_fixture(self, monkeypatch):
        import importlib
        import prompts.loader as loader_mod
        from prompts.loader import _reset_cache

        # Reload agents.prompt fresh to avoid cross-test module identity issues
        for _k in list(sys.modules.keys()):
            if _k == "agents.prompt":
                del sys.modules[_k]
        import agents.prompt as _prompt_mod_fresh
        get_react_system_prompt = _prompt_mod_fresh.get_react_system_prompt

        # Captured baseline fixture from Step 1 before wording updates
        old_system = (
            "## IDENTITY\n"
            "You are VoxelBox Explore Assistant — BrainSightAI's official AI companion.\n"
            "## ABSOLUTE HARD RULES — NEVER BREAK THESE\n"
            "1. Never invent BrainSightAI product details"
        )
        old_react = (
            "## AUTONOMOUS AGENT BEHAVIOUR\n"
            "You are an autonomous reasoning agent.\n"
            "**Critical output rules — follow these exactly:**"
        )
        expected_combined = (
            "\n\n" + "\n\n".join([old_system.strip(), old_react.strip()]) + "\n"
        )

        _reset_cache()
        def mock_load(name):
            if name == "system.md":
                return old_system
            if name == "react_instructions.md":
                return old_react
            raise loader_mod.PromptNotFoundError(name)

        monkeypatch.setattr(loader_mod, "load_prompt", mock_load)
        monkeypatch.setattr(_prompt_mod_fresh, "load_prompt", mock_load)

        actual_combined = get_react_system_prompt()
        assert actual_combined == expected_combined
        _reset_cache()


class TestPromptLoader:
    """Test 2: load_prompt missing file raises PromptNotFoundError; prompt_version is 12 hex chars."""

    def test_missing_prompt_raises(self):
        from prompts.loader import load_prompt, PromptNotFoundError, _reset_cache
        _reset_cache()
        with pytest.raises(PromptNotFoundError):
            load_prompt("nonexistent_prompt_file_12345.md")

    def test_prompt_version_hex_and_stable(self, tmp_path, monkeypatch):
        import prompts.loader as loader_mod
        loader_mod._reset_cache()

        # Create temporary prompts directory
        p1 = tmp_path / "system.md"
        p2 = tmp_path / "react_instructions.md"
        p1.write_text("System prompt v1", encoding="utf-8")
        p2.write_text("React instructions v1", encoding="utf-8")

        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_path)

        v1 = loader_mod.prompt_version()
        assert len(v1) == 12
        assert all(c in "0123456789abcdef" for c in v1)

        # Stable across multiple calls
        assert loader_mod.prompt_version() == v1

        # Changes when file content changes
        loader_mod._reset_cache()
        p1.write_text("System prompt v2 - modified text", encoding="utf-8")
        v2 = loader_mod.prompt_version()
        assert len(v2) == 12
        assert v2 != v1

        loader_mod._reset_cache()


class TestConfigYamlUpdated:
    """Test 3: config.yaml has no system_prompt key, and has prompts.system / prompts.react_instructions."""

    def test_config_yaml_has_prompts_not_system_prompt(self):
        import yaml
        cfg_path = backend_dir / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "system_prompt" not in cfg
        assert "prompts" in cfg
        assert cfg["prompts"]["system"] == "system.md"
        assert cfg["prompts"]["react_instructions"] == "react_instructions.md"


class TestSystemPromptWording:
    """Test 4: prompts/system.md verification."""

    def test_system_prompt_wording(self):
        sys_path = backend_dir / "prompts" / "system.md"
        assert sys_path.is_file()
        text = sys_path.read_text(encoding="utf-8")

        # Removed
        assert "ground truth" not in text
        assert "No citation brackets" not in text

        # Added / Preserved
        assert "TOOLS ARE THE SOURCE OF TRUTH" in text
        assert "[Source: <Document Title>, p. <page>]" in text
        assert "## ROLE BOUNDARY" in text
        assert "9. Never give medical advice, diagnoses, or clinical interpretation of scans" in text


class TestCitationMeta:
    """Test 5: citation_meta helper tests."""

    def test_citation_meta_cases(self, monkeypatch):
        citation_meta = getattr(sys.modules.get("vector_db.retriever"), "citation_meta", ret_mod.citation_meta)

        # Case 1: page_label "5" -> 5
        doc1 = {"metadata": {"source": "Doc.pdf", "page_label": "5"}}
        _, p1 = citation_meta(doc1)
        assert p1 == 5

        # Case 2: page 4 (no label) -> 5
        doc2 = {"metadata": {"source": "Doc.pdf", "page": 4}}
        _, p2 = citation_meta(doc2)
        assert p2 == 5

        # Case 3: no page -> None
        doc3 = {"metadata": {"source": "Doc.txt"}}
        _, p3 = citation_meta(doc3)
        assert p3 is None

        # Case 4: title from citations.titles mapping
        from config import CitationsConfig
        test_cfg = CitationsConfig(
            verify=True,
            titles={"Custom_Filename.pdf": "BrainSight Custom Protocol"}
        )
        monkeypatch.setattr("config.get_citations_config", lambda: test_cfg)
        doc4 = {"metadata": {"source": "Custom_Filename.pdf", "page": 2}}
        t4, p4 = citation_meta(doc4)
        assert t4 == "BrainSight Custom Protocol"
        assert p4 == 3

        # Case 5: title from filename stem
        doc5 = {"metadata": {"source": "DTI_Acquisition_Protocol.pdf", "page": 0}}
        t5, p5 = citation_meta(doc5)
        assert t5 == "DTI Acquisition Protocol"
        assert p5 == 1


class TestRetrieverContextAndSources:
    """Test 6: retriever context contains [Document: DTI Acquisition Protocol | p. 5], and sources include page."""

    def test_retriever_format(self):
        # Always use the ret_mod loaded at this module's collection time
        # (which has VectorStore = MagicMock), not whatever sys.modules may hold
        # after cross-test module interference.
        DocumentRetriever = ret_mod.DocumentRetriever

        doc = types.SimpleNamespace(
            page_content="FA values in tractography.",
            metadata={
                "source": "DTI_Acquisition_Protocol.pdf",
                "page": 4,  # -> page 5
                "retrieval_method": "vector",
                "retrieval_score": 0.95,
                "url": "#",
            }
        )

        mock_store = MagicMock()
        mock_store.config = types.SimpleNamespace(
            vector_db=types.SimpleNamespace(
                search=types.SimpleNamespace(k=3, score_threshold=0.5)
            )
        )
        mock_store.search_hybrid.return_value = [doc]

        retriever = DocumentRetriever()
        retriever.vector_store = mock_store

        res = retriever.retrieve_context("FA tractography")
        assert "[Document: DTI Acquisition Protocol | p. 5]" in res["context"]
        assert "FA values in tractography." in res["context"]

        assert len(res["sources"]) == 1
        s = res["sources"][0]
        assert s["title"] == "DTI Acquisition Protocol"
        assert s["page"] == 5


class TestVerifyCitations:
    """Test 7: verify_citations unit tests."""

    def test_verify_citations(self):
        from services.citations import verify_citations

        sources = [
            {"title": "DTI Acquisition Protocol", "page": 5},
            {"title": "System Architecture Guide", "page": None},
        ]

        # Valid title + page is kept
        t1 = "According to specs [Source: DTI Acquisition Protocol, p. 5], slice thickness is 2mm."
        c1 = verify_citations(t1, sources)
        assert c1.valid == 1
        assert c1.invalid == 0
        assert c1.missing_page == 0
        assert "[Source: DTI Acquisition Protocol, p. 5]" in c1.text

        # Valid title with no page is kept (missing_page tracked)
        t2 = "The architecture [Source: System Architecture Guide] uses microservices."
        c2 = verify_citations(t2, sources)
        assert c2.valid == 1
        assert c2.invalid == 0
        assert c2.missing_page == 1
        assert "[Source: System Architecture Guide]" in c2.text

        # Wrong page: title is valid, page removed, missing_page tracked
        t3 = "Parameters [Source: DTI Acquisition Protocol, p. 99] were set."
        c3 = verify_citations(t3, sources)
        assert c3.valid == 1
        assert c3.invalid == 0
        assert c3.missing_page == 1
        assert "[Source: DTI Acquisition Protocol]" in c3.text
        assert "p. 99" not in c3.text

        # Unknown document is removed
        t4 = "Fake facts [Source: Hallucinated Manual, p. 1] here."
        c4 = verify_citations(t4, sources)
        assert c4.valid == 0
        assert c4.invalid == 1
        assert "Fake facts here." in c4.text

        # Case-insensitive title works
        t5 = "Parameters [Source: dti acquisition protocol, p. 5]."
        c5 = verify_citations(t5, sources)
        assert c5.valid == 1
        assert c5.invalid == 0
        assert "[Source: dti acquisition protocol, p. 5]" in c5.text

        # Multiple citations handled
        t6 = "Real [Source: DTI Acquisition Protocol, p. 5] and fake [Source: Ghost Manual, p. 2]."
        c6 = verify_citations(t6, sources)
        assert c6.valid == 1
        assert c6.invalid == 1
        assert "[Source: DTI Acquisition Protocol, p. 5]" in c6.text
        assert "Ghost Manual" not in c6.text

        # Forms "p 5" and "p.5"
        t7 = "Test p [Source: DTI Acquisition Protocol, p 5] and dot [Source: DTI Acquisition Protocol, p.5]."
        c7 = verify_citations(t7, sources)
        assert c7.valid == 2
        assert c7.invalid == 0

        # Garbage input returns unchanged
        assert verify_citations(None, sources).text is None
        assert verify_citations(12345, sources).text == 12345


class TestChatCitationsIntegration:
    """Test 8: chat() with fake orchestrator citing real and fake docs."""

    def test_chat_filters_fake_citation(self, monkeypatch):
        from memory.store import _reset_memory_store, get_memory_store
        from database.models import FeedbackInteraction
        _reset_memory_store()
        _orch_mock.reset_mock()

        # Orchestrator returns answer with one real and one fake citation
        real_cite = "[Source: DTI Acquisition Protocol, p. 5]"
        fake_cite = "[Source: Imaginary Guide, p. 12]"
        raw_answer = f"Acquisition parameter is 1000 {real_cite}. Also fake parameter {fake_cite}."

        orch_ret = MagicMock()
        orch_ret.content = raw_answer
        orch_ret.error = None
        orch_ret.sources = [
            {"title": "DTI Acquisition Protocol", "page": 5, "url": "#", "content": "..."}
        ]
        orch_ret.tool_calls = []
        orch_ret.react_steps = []
        orch_ret.messages = [
            Message(role="user", content="tell me parameters"),
            Message(role="assistant", content=raw_answer),
        ]
        orch_ret.provider_used = "gemini"
        orch_ret.tokens_used = 50
        _orch_mock.run.return_value = orch_ret

        # Disable medical guardrail to test citations in isolation
        monkeypatch.setattr(
            "routers.chat.get_guardrails_config",
            lambda: types.SimpleNamespace(medical=types.SimpleNamespace(enabled=False, output_check=False))
        )
        # Ensure the chat router uses the mock orchestrator regardless of import order
        monkeypatch.setattr("routers.chat.orchestrator", _orch_mock)

        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        client = _make_client()
        resp = client.post("/api/chat/", json={
            "message": "tell me parameters",
            "session_id": session_id,
            "user_id": "tester",
        })
        assert resp.status_code == 200
        data = resp.json()

        # Response has real citation and NOT fake citation
        assert real_cite in data["response"]
        assert fake_cite not in data["response"]

        # Sources include page
        assert len(data["sources"]) == 1
        assert data["sources"][0]["page"] == 5

        # Memory stores cleaned text
        mem = get_memory_store().load(session_id)
        assert len(mem) >= 2
        last_asst = [m for m in mem if m.role == "assistant"][-1]
        assert real_cite in last_asst.content
        assert fake_cite not in last_asst.content

        # FeedbackInteraction stores cleaned text
        db = SessionLocal()
        try:
            fi = db.query(FeedbackInteraction).filter(FeedbackInteraction.session_id == session_id).first()
            assert fi is not None
            assert real_cite in fi.response
            assert fake_cite not in fi.response
        finally:
            db.close()


class TestHealthEndpointPromptVersion:
    """Test 9: /api/health includes prompt_version."""

    def test_health_includes_prompt_version(self):
        client = _make_client()
        resp = client.get("/api/health/")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt_version" in data
        assert len(data["prompt_version"]) == 12


class TestCitationFixes:
    """Tests for citation title normalization, page tolerance, and diagnostics (Step 5)."""

    def test_01_citation_meta_tm_upload_single_spaces(self):
        citation_meta = getattr(sys.modules.get("vector_db.retriever"), "citation_meta", ret_mod.citation_meta)
        doc = {"metadata": {"source": "TM_ Upload & download script.pdf", "page": 0}}
        title, page = citation_meta(doc)
        assert title == "TM Upload & download script"
        assert "  " not in title

    def test_02_normalize_title(self):
        from services.citations import normalize_title
        raw = "TM  Upload_&_download-script.PDF"
        norm = normalize_title(raw)
        assert norm == "tm upload & download script"

    def test_03_cited_matches_retrieved_double_space(self):
        from services.citations import verify_citations
        sources = [{"title": "TM  Upload & download script", "page": 1}]
        text = "Follow instructions [Source: TM Upload & download script, p. 1]."
        res = verify_citations(text, sources)
        assert res.valid == 1
        assert res.invalid == 0
        assert "[Source: TM Upload & download script, p. 1]" in res.text

    def test_04_case_insensitivity(self):
        from services.citations import verify_citations
        sources = [{"title": "VoxelBox v5.0.3", "page": 2}]
        text = "Details are here [Source: voxelbox v5.0.3, p. 2]."
        res = verify_citations(text, sources)
        assert res.valid == 1
        assert res.invalid == 0
        assert "[Source: voxelbox v5.0.3, p. 2]" in res.text

    def test_05_containment_matching_rule(self):
        from services.citations import verify_citations
        sources = [
            {"title": "BrainSightAI Frequently Asked Questions - Clinicians", "page": 1},
            {"title": "FAQ Document", "page": 1},
        ]
        # Containment with >=3 words matches
        text_kept = "Info [Source: BrainSightAI Frequently Asked Questions, p. 1]."
        res_kept = verify_citations(text_kept, sources)
        assert res_kept.valid == 1
        assert res_kept.invalid == 0
        assert "[Source: BrainSightAI Frequently Asked Questions, p. 1]" in res_kept.text

        # Containment with 1 word (<3 words) does NOT match
        text_dropped = "Info [Source: FAQ, p. 1]."
        res_dropped = verify_citations(text_dropped, sources)
        assert res_dropped.valid == 0
        assert res_dropped.invalid == 1
        assert "[Source: FAQ, p. 1]" not in res_dropped.text

    def test_06_wrong_page_page_removed_citation_kept(self):
        from services.citations import verify_citations
        sources = [{"title": "Neuronavigation", "page": 4}]
        text = "Export steps [Source: Neuronavigation, p. 99] are listed."
        res = verify_citations(text, sources)
        assert res.valid == 1
        assert res.invalid == 0
        assert res.missing_page == 1
        assert "[Source: Neuronavigation]" in res.text
        assert "p. 99" not in res.text

    def test_07_unknown_title_removed(self):
        from services.citations import verify_citations
        sources = [{"title": "Neuronavigation", "page": 4}]
        text = "Fabricated info [Source: Nonexistent Medical Manual, p. 1] end."
        res = verify_citations(text, sources)
        assert res.valid == 0
        assert res.invalid == 1
        assert "[Source: Nonexistent Medical Manual, p. 1]" not in res.text
        assert "Fabricated info end." in res.text

    def test_08_react_instructions_has_checklist(self):
        instructions_path = backend_dir / "prompts" / "react_instructions.md"
        content = instructions_path.read_text(encoding="utf-8")
        assert "Before giving your final answer:" in content
        assert "[Source: <title>, p. <page>]" in content
        assert "At least one citation is required" in content

    def test_09_diagnostics_logging(self, caplog):
        import logging
        from services.citations import verify_citations
        sources = [{"title": "VoxelBox Guide", "page": 1}]
        secret_answer = "Secret internal patient answer text that must not be logged."
        text = f"{secret_answer} [Source: Unknown Guide, p. 5]"

        with caplog.at_level(logging.INFO, logger="services.citations"):
            verify_citations(text, sources, message_id="msg_diag_123")

        assert "retrieved_titles=" in caplog.text
        assert "citations message_id=msg_diag_123" in caplog.text
        assert "citation removed message_id=msg_diag_123" in caplog.text
        assert "cited_title=\"Unknown Guide\"" in caplog.text
        assert secret_answer not in caplog.text
