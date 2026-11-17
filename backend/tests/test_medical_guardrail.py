"""
Pytest suite for the medical-advice guardrail (V2 item 4).
No network calls — all external calls are monkeypatched.
No imports from llm/__init__.py (loads every provider SDK).
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
from typing import List, Optional
from unittest.mock import MagicMock, patch, call

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

# stub llm.factory
_factory_mod = types.ModuleType("llm.factory")
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

# stub services (orchestrator + suggestions) so routers/chat.py imports cleanly
services_dir = str(backend_dir / "services")
_svc_mod = types.ModuleType("services")
_svc_mod.__path__ = [services_dir]
_orch_mock = MagicMock()
_sug_mock = MagicMock()
_sug_mock.build.return_value = []
_svc_mod.orchestrator = _orch_mock
_svc_mod.suggestions = _sug_mock
sys.modules["services"] = _svc_mod

if "services.job_status" not in sys.modules:
    spec_job = importlib.util.spec_from_file_location("services.job_status", backend_dir / "services" / "job_status.py")
    job_mod = importlib.util.module_from_spec(spec_job)
    spec_job.loader.exec_module(job_mod)
    sys.modules["services.job_status"] = job_mod
    _svc_mod.job_status = job_mod

# bare routers package (don't execute __init__.py which imports chat router)
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
# 3. Temp working directory with real config.yaml + fresh SQLite DB
# ─────────────────────────────────────────────────────────────────────────────

_tmp = Path(tempfile.mkdtemp(prefix="bsai_guardrail_test_"))
_orig_cwd = os.getcwd()
os.chdir(_tmp)
shutil.copy(backend_dir / "config.yaml", _tmp / "config.yaml")
(_tmp / "data" / "database").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("JWT_SECRET", "test-secret-at-least-32-chars-long-abcdef123456")
# Remove GOOGLE_API_KEY so tests are truly offline (individual tests override as needed)
os.environ.pop("GOOGLE_API_KEY", None)

from database.connection import init_db, SessionLocal
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build a minimal FastAPI test app with the chat router
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.chat import router as chat_router
from guardrails.medical import GuardrailDecision, _reset_guardrail

_app = FastAPI()
_app.include_router(chat_router)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Helpers
# ─────────────────────────────────────────────────────────────────────────────

CANNED = (
    "I am an operational assistant for VoxelBox software and cannot provide "
    "medical advice or evaluate symptoms. Please consult your physician."
)


def _make_client():
    return TestClient(_app, raise_server_exceptions=True)


def _blocked_decision(stage="input"):
    return GuardrailDecision(
        blocked=True,
        category="personal_medical",
        reason="user asking for medical advice",
        latency_ms=12.0,
        error=None,
        stage=stage,
    )


def _allowed_decision(stage="input"):
    return GuardrailDecision(
        blocked=False,
        category="technical",
        reason="technical VoxelBox question",
        latency_ms=10.0,
        error=None,
        stage=stage,
    )


def _error_decision(stage="input", on_block=False):
    return GuardrailDecision(
        blocked=on_block,
        category=None,
        reason="classifier error",
        latency_ms=5.0,
        error="timeout",
        stage=stage,
    )


def _orch_result(content="Here is the answer."):
    r = MagicMock()
    r.content = content
    r.error = None
    r.sources = []
    r.tool_calls = []
    r.react_steps = []
    r.messages = [
        Message(role="user", content="how does it work"),
        Message(role="assistant", content=content),
    ]
    r.provider_used = "gemini"
    r.tokens_used = 100
    return r


def _session():
    return f"ses_{uuid.uuid4().hex[:8]}"


def _post(client, session_id, message="Tell me how the pipeline works"):
    return client.post("/api/chat/", json={
        "message": message,
        "session_id": session_id,
        "user_id": "tester",
    })


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardrailBlocked:
    """Test 1: blocked input → canned response, guardrail field set, no orchestrator call."""

    def test_blocked_input_canned_response(self, monkeypatch):
        from guardrails import get_medical_guardrail
        from memory.store import _reset_memory_store
        _reset_guardrail()
        _reset_memory_store()
        _orch_mock.reset_mock()

        gr = MagicMock()
        gr.check_input.return_value = _blocked_decision()
        gr.check_output.return_value = _allowed_decision("output")
        monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: gr)

        client = _make_client()
        resp = client.post("/api/chat/", json={
            "message": "My father has a fever, what should he take?",
            "session_id": _session(),
            "user_id": "tester",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == CANNED
        assert data["guardrail"] == "medical_advice"
        assert data["provider_used"] == "guardrail"
        _orch_mock.run.assert_not_called()


class TestGuardrailAllowed:
    """Test 2: allowed input → orchestrator called, guardrail is None."""

    def test_allowed_input_orchestrator_called(self, monkeypatch):
        from memory.store import _reset_memory_store
        _reset_guardrail()
        _reset_memory_store()
        _orch_mock.reset_mock()
        _orch_mock.run.return_value = _orch_result()
        monkeypatch.setattr("routers.chat.orchestrator", _orch_mock)

        gr = MagicMock()
        gr.check_input.return_value = _allowed_decision()
        gr.check_output.return_value = _allowed_decision("output")
        monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: gr)

        client = _make_client()
        resp = _post(client, _session())
        assert resp.status_code == 200
        data = resp.json()
        assert data["guardrail"] is None
        assert data["provider_used"] == "gemini"
        _orch_mock.run.assert_called_once()


class TestGuardrailMemory:
    """Test 3: blocked turn is saved to memory as user + canned assistant."""

    def test_blocked_saved_to_memory(self, monkeypatch):
        from memory.store import _reset_memory_store, get_memory_store
        _reset_guardrail()
        _reset_memory_store()
        _orch_mock.reset_mock()

        session = _session()
        gr = MagicMock()
        gr.check_input.return_value = _blocked_decision()
        monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: gr)

        client = _make_client()
        client.post("/api/chat/", json={
            "message": "Am I at risk of a stroke?",
            "session_id": session,
            "user_id": "tester",
        })

        mem = get_memory_store().load(session)
        assert len(mem) >= 2
        roles = [m.role for m in mem]
        assert "user" in roles
        assert "assistant" in roles
        # canned reply is in memory
        assert any(m.content == CANNED for m in mem)
        # unsafe answer (orchestrator never ran) is NOT in memory
        assert not any(m.content == "orchestrator answer" for m in mem)


class TestGuardrailDBRecords:
    """Test 4: blocked turn writes FeedbackInteraction (canned) and GuardrailEvent (blocked); no audit."""

    def test_db_records_on_block(self, monkeypatch):
        from memory.store import _reset_memory_store
        from database.models import FeedbackInteraction as FI, GuardrailEvent as GE
        _reset_guardrail()
        _reset_memory_store()
        _orch_mock.reset_mock()

        session = _session()
        msg = "What does this lesion mean for the patient?"
        gr = MagicMock()
        gr.check_input.return_value = _blocked_decision()
        monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: gr)

        # Spy on _write_audit
        audit_called = []
        import routers.chat as chat_mod
        orig_audit = chat_mod._write_audit
        def spy_audit(*a, **kw):
            audit_called.append(True)
            return orig_audit(*a, **kw)
        monkeypatch.setattr(chat_mod, "_write_audit", spy_audit)

        client = _make_client()
        resp = client.post("/api/chat/", json={
            "message": msg,
            "session_id": session,
            "user_id": "tester",
        })
        assert resp.status_code == 200

        db = SessionLocal()
        try:
            fi = db.query(FI).filter(FI.session_id == session).first()
            assert fi is not None
            assert fi.response == CANNED
            assert fi.provider_used == "guardrail"

            ge = db.query(GE).filter(GE.session_id == session, GE.stage == "input").first()
            assert ge is not None
            assert ge.decision == "blocked"
            assert ge.category == "personal_medical"
        finally:
            db.close()

        assert len(audit_called) == 0, "_write_audit must NOT be called for blocked inputs"


class TestGuardrailErrorPolicy:
    """Test 5: classifier error respects on_error policy; both log decision='error'."""

    def test_error_allow(self, monkeypatch):
        from memory.store import _reset_memory_store
        from database.models import GuardrailEvent as GE
        _reset_guardrail()
        _reset_memory_store()
        _orch_mock.reset_mock()
        _orch_mock.run.return_value = _orch_result()
        monkeypatch.setattr("routers.chat.orchestrator", _orch_mock)

        session = _session()
        gr = MagicMock()
        gr.check_input.return_value = _error_decision(on_block=False)
        gr.check_output.return_value = _allowed_decision("output")
        monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: gr)

        client = _make_client()
        resp = _post(client, session)
        assert resp.status_code == 200
        # allow → orchestrator runs
        _orch_mock.run.assert_called_once()

        db = SessionLocal()
        try:
            ge = db.query(GE).filter(GE.session_id == session, GE.stage == "input").first()
            assert ge is not None
            assert ge.decision == "error"
        finally:
            db.close()

    def test_error_block(self, monkeypatch):
        from memory.store import _reset_memory_store
        from database.models import GuardrailEvent as GE
        _reset_guardrail()
        _reset_memory_store()
        _orch_mock.reset_mock()

        session = _session()
        gr = MagicMock()
        gr.check_input.return_value = _error_decision(on_block=True)
        monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: gr)

        # Temporarily switch on_error to "block" via config override
        from config import get_config, _config as _cfg_ref
        import config as config_mod
        orig_cfg = config_mod._config

        # patch get_guardrails_config
        import routers.chat as chat_mod
        orig_gr_cfg = chat_mod.get_guardrails_config
        def _patched_gr_cfg():
            from config import get_config
            cfg = get_config()
            from config import MedicalGuardrailConfig, GuardrailsConfig
            return GuardrailsConfig(
                medical=MedicalGuardrailConfig(
                    enabled=True,
                    model="gemini-3.5-flash-lite",
                    timeout_s=3.0,
                    on_error="block",
                    output_check=False,
                    canned_response=CANNED,
                )
            )
        monkeypatch.setattr(chat_mod, "get_guardrails_config", _patched_gr_cfg)

        client = _make_client()
        resp = _post(client, session)
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == CANNED
        assert data["guardrail"] == "medical_advice"
        _orch_mock.run.assert_not_called()

        db = SessionLocal()
        try:
            ge = db.query(GE).filter(GE.session_id == session, GE.stage == "input").first()
            assert ge is not None
            assert ge.decision == "error"
        finally:
            db.close()


class TestGuardrailDisabled:
    """Test 6: enabled=false → check_input never called."""

    def test_disabled_no_check(self, monkeypatch):
        from memory.store import _reset_memory_store
        _reset_guardrail()
        _reset_memory_store()
        _orch_mock.reset_mock()
        _orch_mock.run.return_value = _orch_result()
        monkeypatch.setattr("routers.chat.orchestrator", _orch_mock)

        gr = MagicMock()
        gr.check_input.return_value = _blocked_decision()
        monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: gr)

        import routers.chat as chat_mod
        def _disabled_gr_cfg():
            from config import MedicalGuardrailConfig, GuardrailsConfig
            return GuardrailsConfig(
                medical=MedicalGuardrailConfig(
                    enabled=False,
                    model="gemini-3.5-flash-lite",
                    timeout_s=3.0,
                    on_error="allow",
                    output_check=False,
                    canned_response=CANNED,
                )
            )
        monkeypatch.setattr(chat_mod, "get_guardrails_config", _disabled_gr_cfg)

        client = _make_client()
        resp = _post(client, _session())
        assert resp.status_code == 200
        gr.check_input.assert_not_called()
        _orch_mock.run.assert_called_once()


class TestGuardrailOutputCheck:
    """Test 7: output_check=true + unsafe answer → canned, memory clean, event logged."""

    def test_unsafe_output_replaced(self, monkeypatch):
        from memory.store import _reset_memory_store, get_memory_store
        from database.models import GuardrailEvent as GE
        _reset_guardrail()
        _reset_memory_store()
        _orch_mock.reset_mock()

        unsafe_answer = "The scan shows a malignant tumor; surgery is required immediately."
        _orch_mock.run.return_value = _orch_result(content=unsafe_answer)
        monkeypatch.setattr("routers.chat.orchestrator", _orch_mock)

        gr = MagicMock()
        gr.check_input.return_value = _allowed_decision("input")
        gr.check_output.return_value = GuardrailDecision(
            blocked=True,
            category="clinical_interpretation",
            reason="answer interprets scan clinically",
            latency_ms=15.0,
            error=None,
            stage="output",
        )
        monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: gr)

        import routers.chat as chat_mod
        def _output_check_cfg():
            from config import MedicalGuardrailConfig, GuardrailsConfig
            return GuardrailsConfig(
                medical=MedicalGuardrailConfig(
                    enabled=True,
                    model="gemini-3.5-flash-lite",
                    timeout_s=3.0,
                    on_error="allow",
                    output_check=True,
                    canned_response=CANNED,
                )
            )
        monkeypatch.setattr(chat_mod, "get_guardrails_config", _output_check_cfg)

        session = _session()
        client = _make_client()
        resp = client.post("/api/chat/", json={
            "message": "What does the scan show?",
            "session_id": session,
            "user_id": "tester",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == CANNED
        assert data["guardrail"] == "medical_advice"

        # Memory must NOT contain the unsafe answer
        mem = get_memory_store().load(session)
        assert not any(m.content == unsafe_answer for m in mem)
        assert any(m.content == CANNED for m in mem)

        # GuardrailEvent with stage="output"
        db = SessionLocal()
        try:
            ge = db.query(GE).filter(GE.session_id == session, GE.stage == "output").first()
            assert ge is not None
            assert ge.decision == "blocked"
        finally:
            db.close()


class TestGuardrailOutputCheckDisabled:
    """Test 8: output_check=false → check_output never called."""

    def test_check_output_not_called(self, monkeypatch):
        from memory.store import _reset_memory_store
        _reset_guardrail()
        _reset_memory_store()
        _orch_mock.reset_mock()
        _orch_mock.run.return_value = _orch_result()
        monkeypatch.setattr("routers.chat.orchestrator", _orch_mock)

        gr = MagicMock()
        gr.check_input.return_value = _allowed_decision("input")
        monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: gr)

        # output_check is False by default; don't override config
        client = _make_client()
        resp = _post(client, _session())
        assert resp.status_code == 200
        gr.check_output.assert_not_called()


class TestBuildContents:
    """Test 9: contents builder unit tests (no network)."""

    def _get_guardrail(self):
        """Return a fresh MedicalGuardrail instance with real config."""
        _reset_guardrail()
        from guardrails.medical import MedicalGuardrail
        from config import MedicalGuardrailConfig
        cfg = MedicalGuardrailConfig(
            enabled=True,
            model="gemini-3.5-flash-lite",
            timeout_s=3.0,
            history_turns=2,
            on_error="allow",
            output_check=False,
            canned_response=CANNED,
        )
        gr = MedicalGuardrail(cfg=cfg)
        return gr

    def _make_msg(self, role, content="", tool_calls=None):
        return Message(role=role, content=content, tool_calls=tool_calls)

    def test_empty_history(self):
        from google.genai import types
        gr = self._get_guardrail()
        contents = gr._build_contents("hello", [])
        assert len(contents) == 1
        assert contents[-1].role == "user"
        assert contents[-1].parts[0].text == "hello"

    def test_only_user_assistant_included(self):
        from google.genai import types
        gr = self._get_guardrail()
        history = [
            self._make_msg("user", "first question"),
            self._make_msg("tool", "tool result"),        # must be skipped
            self._make_msg("assistant", "first answer"),
        ]
        contents = gr._build_contents("second question", history)
        roles = [c.role for c in contents]
        assert "tool" not in roles
        assert contents[-1].role == "user"
        assert contents[-1].parts[0].text == "second question"

    def test_tool_call_only_assistant_skipped(self):
        """Assistant messages with no content (only tool_calls) are skipped."""
        from google.genai import types
        gr = self._get_guardrail()
        tc = ToolCall(id="1", name="get_job", arguments={})
        history = [
            self._make_msg("user", "run it"),
            self._make_msg("assistant", "", tool_calls=[tc]),  # no text → skip
            self._make_msg("assistant", "here is the result"),
        ]
        contents = gr._build_contents("next", history)
        # tool-only assistant turn should not appear
        model_turns = [c for c in contents if c.role == "model"]
        assert all(c.parts[0].text != "" for c in model_turns)

    def test_at_most_history_turns_pairs(self):
        from google.genai import types
        gr = self._get_guardrail()
        # history_turns=2 → at most 2 pairs = 4 messages
        history = []
        for i in range(5):
            history.append(self._make_msg("user", f"q{i}"))
            history.append(self._make_msg("assistant", f"a{i}"))
        contents = gr._build_contents("new q", history)
        # Last item is always the current message
        assert contents[-1].parts[0].text == "new q"
        # At most 2*2+1 = 5 items (2 pairs + current)
        assert len(contents) <= 5

    def test_last_content_is_always_user(self):
        from google.genai import types
        gr = self._get_guardrail()
        history = [
            self._make_msg("user", "q1"),
            self._make_msg("assistant", "a1"),
            self._make_msg("assistant", "a2"),  # trailing model turn → removed
        ]
        contents = gr._build_contents("current", history)
        assert contents[-1].role == "user"


class TestMissingApiKey:
    """Test 10: missing GOOGLE_API_KEY → check_input returns error decision, does not raise."""

    def test_missing_key_returns_error(self):
        _reset_guardrail()
        orig_key = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            from guardrails.medical import MedicalGuardrail, GuardrailDecision
            from config import MedicalGuardrailConfig
            cfg = MedicalGuardrailConfig(
                enabled=True,
                model="gemini-3.5-flash-lite",
                timeout_s=3.0,
                on_error="allow",
                output_check=False,
                canned_response=CANNED,
            )
            gr = MedicalGuardrail(cfg=cfg)
            decision = gr.check_input("Am I sick?", [])
            assert isinstance(decision, GuardrailDecision)
            assert decision.error is not None
            assert decision.blocked is False   # on_error="allow"
        finally:
            if orig_key is not None:
                os.environ["GOOGLE_API_KEY"] = orig_key
            _reset_guardrail()


class TestConfigYamlUpdated:
    """Test 11: config.yaml / prompts/system.md has ROLE BOUNDARY and hard rule 9; disclaimer removed."""

    def test_disclaimer_removed(self):
        prompts_file = backend_dir / "prompts" / "system.md"
        cfg_file = backend_dir / "config.yaml"
        text = prompts_file.read_text(encoding="utf-8") if prompts_file.is_file() else cfg_file.read_text(encoding="utf-8")
        assert "Not medical advice — consult your clinician" not in text

    def test_role_boundary_present(self):
        prompts_file = backend_dir / "prompts" / "system.md"
        cfg_file = backend_dir / "config.yaml"
        text = prompts_file.read_text(encoding="utf-8") if prompts_file.is_file() else cfg_file.read_text(encoding="utf-8")
        assert "## ROLE BOUNDARY" in text
        assert "technical copilot for VoxelBox" in text

    def test_hard_rule_9_present(self):
        prompts_file = backend_dir / "prompts" / "system.md"
        cfg_file = backend_dir / "config.yaml"
        text = prompts_file.read_text(encoding="utf-8") if prompts_file.is_file() else cfg_file.read_text(encoding="utf-8")
        assert "9. Never give medical advice" in text


class TestExistingDatabaseSchemaCompatibility:
    """Fix 4 required test: verify compatibility with existing production feedback.db schema."""

    def test_committed_feedback_db_compatibility(self, monkeypatch, tmp_path):
        from database.connection import init_db, SessionLocal, engine
        from database.models import FeedbackInteraction as FI, GuardrailEvent as GE
        from memory.store import _reset_memory_store

        # 1. Copy the committed backend/data/database/feedback.db to temp dir
        committed_db = backend_dir / "data" / "database" / "feedback.db"
        assert committed_db.is_file(), f"Committed feedback.db not found at {committed_db}"

        target_db_dir = tmp_path / "data" / "database"
        target_db_dir.mkdir(parents=True, exist_ok=True)
        target_db = target_db_dir / "feedback.db"
        shutil.copy(committed_db, target_db)
        shutil.copy(backend_dir / "config.yaml", tmp_path / "config.yaml")

        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            engine.dispose()
            # 2. Run init_db() against the copied database
            init_db()

            _reset_guardrail()
            _reset_memory_store()

            # 3. Run a chat request (fake orchestrator) that writes a FeedbackInteraction
            _orch_mock.reset_mock()
            _orch_mock.run.return_value = _orch_result("Technical info about TR/TE.")
            monkeypatch.setattr("routers.chat.orchestrator", _orch_mock)

            mock_gr = MagicMock()
            mock_gr.check_input.return_value = GuardrailDecision(
                blocked=False,
                category="technical",
                reason="technical question",
                latency_ms=10.0,
                error=None,
                stage="input",
            )
            mock_gr.check_output.return_value = GuardrailDecision(
                blocked=False,
                category="safe",
                reason="safe output",
                latency_ms=5.0,
                error=None,
                stage="output",
            )
            monkeypatch.setattr("routers.chat.get_medical_guardrail", lambda: mock_gr)

            test_client = _make_client()
            resp_allowed = test_client.post(
                "/api/chat/",
                json={"message": "What is the recommended TR for T1?", "session_id": "compat-test-sess-1"},
            )
            assert resp_allowed.status_code == 200, resp_allowed.text

            # 4. Run a blocked request that writes a GuardrailEvent
            mock_gr.check_input.return_value = GuardrailDecision(
                blocked=True,
                category="personal_medical",
                reason="medical query",
                latency_ms=12.0,
                error=None,
                stage="input",
            )

            resp_blocked = test_client.post(
                "/api/chat/",
                json={"message": "Do I have brain cancer?", "session_id": "compat-test-sess-2"},
            )
            assert resp_blocked.status_code == 200, resp_blocked.text
            assert "operational assistant" in resp_blocked.json()["response"]

            # 5. Assert no errors and that the rows exist
            db = SessionLocal()
            try:
                fi_row = db.query(FI).filter(FI.session_id == "compat-test-sess-1").first()
                assert fi_row is not None
                assert fi_row.question == "What is the recommended TR for T1?"

                ge_row = db.query(GE).filter(GE.session_id == "compat-test-sess-2").first()
                assert ge_row is not None
                assert ge_row.decision == "blocked"
                assert ge_row.category == "personal_medical"
            finally:
                db.close()

        finally:
            engine.dispose()
            os.chdir(orig_cwd)
