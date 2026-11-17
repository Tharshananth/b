"""
Pytest suite for Latency and Cost optimizations (V2 item 5).
Covers all 15 specification checks:
1. match_greeting passes ALL 41 GREETING_CASES from pretest_latency_cost.py
2. Instant greetings fast_path
3. Question not fast-pathed
4. Gatekeeper out_of_scope routing
5. Gatekeeper technical routing
6. Agent 2-round cap + forced final answer with tool_choice="none"
7. Forced final call failure fallback (turn not saved)
8. ReActAgent default max_iterations from config (2)
9. Gemini provider tool_choice config builder
10. cached_tokens extraction and summation
11. Stable system prompt + react instructions rules
12. Parallel gatekeeper with blocked guardrail
13. Parallel gatekeeper with allowed guardrail
14. All 6 return paths write exactly one RequestMetric row
15. Schema safety against feedback.db copy
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config import get_config, get_fast_path_config
from database.models import Base, FeedbackInteraction, RequestMetric, DataAccessLog
from database.connection import SessionLocal
from fast_path import match_greeting
from guardrails.medical import GuardrailDecision
from llm.base import Message, LLMResponse, ToolCall
from agents.state import ReActResult, ReActStep, ToolInvocation
from agents.react_agent import ReActAgent
from agents.prompt import get_react_system_prompt
from routers.chat import ChatRequest, chat


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def test_db():
    """In-memory SQLite DB session for testing routers and metrics."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# 1. match_greeting passes ALL 41 GREETING_CASES
# ─────────────────────────────────────────────────────────────────────────────

def test_01_match_greeting_all_cases():
    script_path = BACKEND / "scripts" / "pretest_latency_cost.py"
    spec = importlib.util.spec_from_file_location("pretest_latency_cost", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    greeting_cases = mod.GREETING_CASES

    assert len(greeting_cases) == 41
    for msg, expected in greeting_cases:
        got = match_greeting(msg)
        assert got == expected, f"Failed on {msg!r}: expected {expected}, got {got}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Greeting fast_path
# ─────────────────────────────────────────────────────────────────────────────

def test_02_greeting_fast_path(test_db):
    req = ChatRequest(message="Hi", session_id="session_fast_02")
    with patch("routers.chat.get_medical_guardrail") as mock_gr, \
         patch("routers.chat.orchestrator.run") as mock_orch:

        resp = chat(req, db=test_db)

        mock_gr.assert_not_called()
        mock_orch.assert_not_called()

        cfg_reply = get_fast_path_config().replies.greeting
        assert resp.response == cfg_reply
        assert resp.provider_used == "fast_path"
        assert resp.guardrail is None

        # Check memory
        from memory import get_memory_store
        history = get_memory_store().load("session_fast_02")
        assert len(history) >= 2
        assert history[-2].content == "Hi"
        assert history[-1].content == cfg_reply

        # Check request_metrics row
        metrics = test_db.query(RequestMetric).filter_by(session_id="session_fast_02").all()
        assert len(metrics) == 1
        m = metrics[0]
        assert m.path == "fast_path"
        assert m.llm_calls == 0
        assert m.provider == "fast_path"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Question with greeting prefix is NOT fast-pathed
# ─────────────────────────────────────────────────────────────────────────────

def test_03_question_not_fast_pathed():
    assert match_greeting("hi, what's the status of patient 2013?") is None
    assert match_greeting("hello can you help me upload a scan") is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gatekeeper category out_of_scope
# ─────────────────────────────────────────────────────────────────────────────

def test_04_gatekeeper_out_of_scope_route(test_db):
    req = ChatRequest(message="Write me a poem about the ocean", session_id="session_oos_04")
    mock_decision = GuardrailDecision(
        blocked=False,
        category="out_of_scope",
        reason="Poem is out of scope",
        latency_ms=12.5,
        error=None,
        stage="input",
        model="gemini-3.5-flash-lite",
    )

    with patch("routers.chat.get_medical_guardrail") as mock_gr, \
         patch("routers.chat.orchestrator.run") as mock_orch:
        mock_gr.return_value.check_input.return_value = mock_decision

        resp = chat(req, db=test_db)

        mock_orch.assert_not_called()
        cfg_oos = get_fast_path_config().replies.out_of_scope
        assert resp.response == cfg_oos
        assert resp.provider_used == "guardrail_route"

        metrics = test_db.query(RequestMetric).filter_by(session_id="session_oos_04").all()
        assert len(metrics) == 1
        assert metrics[0].path == "guardrail_route"
        assert metrics[0].llm_calls == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gatekeeper category technical -> orchestrator called
# ─────────────────────────────────────────────────────────────────────────────

def test_05_gatekeeper_technical_calls_orchestrator(test_db):
    req = ChatRequest(message="What b-value does DTI use?", session_id="session_tech_05")
    mock_decision = GuardrailDecision(
        blocked=False,
        category="technical",
        reason="DTI technical question",
        latency_ms=15.0,
        error=None,
        stage="input",
        model="gemini-3.5-flash-lite",
    )
    mock_res = ReActResult(
        content="DTI uses b=1000 s/mm2.",
        provider_used="gemini",
        tokens_used=120,
        tool_calls=[],
        sources=[],
        messages=[
            Message(role="user", content=req.message),
            Message(role="assistant", content="DTI uses b=1000 s/mm2."),
        ],
        react_steps=[],
        total_execution_time_ms=500.0,
        llm_calls=1,
    )

    with patch("routers.chat.get_medical_guardrail") as mock_gr, \
         patch("routers.chat.orchestrator.run", return_value=mock_res) as mock_orch:
        mock_gr.return_value.check_input.return_value = mock_decision

        resp = chat(req, db=test_db)
        mock_orch.assert_called_once()
        assert resp.response == "DTI uses b=1000 s/mm2."


# ─────────────────────────────────────────────────────────────────────────────
# 6. Agent 2-round cap with forced final answer
# ─────────────────────────────────────────────────────────────────────────────

def test_06_agent_forced_final_answer(test_db):
    call_log = []

    def fake_generate_with_fallback(*args, **kwargs):
        call_log.append(kwargs)
        if len(call_log) == 1:
            return LLMResponse(
                content="Step 1 thinking",
                model="gemini-3.6-flash",
                provider="gemini",
                tool_calls=[ToolCall(id="c1", name="get_job_status", arguments={"patient_id": "1"})],
            )
        elif len(call_log) == 2:
            return LLMResponse(
                content="Step 2 thinking",
                model="gemini-3.6-flash",
                provider="gemini",
                tool_calls=[ToolCall(id="c2", name="search_knowledge_base", arguments={"query": "dti"})],
            )
        else:
            # 3rd call is forced final answer
            return LLMResponse(
                content="Here is the final consolidated answer.",
                model="gemini-3.6-flash",
                provider="gemini",
                tool_calls=None,
            )

    fake_factory = MagicMock()
    fake_factory.generate_with_fallback = fake_generate_with_fallback

    # Mock tools
    fake_registry = MagicMock()
    fake_tool = MagicMock()
    fake_tool.execute.return_value = ("tool output", ToolInvocation(name="tool", arguments={}, ok=True, summary="ok"), [])
    fake_registry.get_tool.return_value = fake_tool
    fake_registry.get_all_tool_specs.return_value = []

    agent = ReActAgent(registry=fake_registry, max_iterations=2)
    agent.factory = fake_factory

    result = agent.run("Check status and protocol")

    assert len(call_log) == 3
    # Check that 3rd call used tool_choice="none" and same model
    assert call_log[2].get("tool_choice") == "none"
    assert call_log[2].get("model") == "gemini-3.6-flash"

    assert result.content == "Here is the final consolidated answer."
    assert result.error is None
    assert result.capped is True
    assert result.llm_calls == 3

    # Check synthetic message not in result.messages
    for m in result.messages:
        assert "Step limit reached" not in m.content

    # Now verify chat() saving this turn to memory and metric path="agent_capped"
    req = ChatRequest(message="Check status and protocol", session_id="session_capped_06")
    with patch("routers.chat.get_medical_guardrail") as mock_gr, \
         patch("routers.chat.orchestrator.run", return_value=result):
        mock_gr.return_value.check_input.return_value = GuardrailDecision(
            blocked=False, category="technical", reason="ok", latency_ms=10.0, error=None, stage="input", model="gemini-3.5-flash-lite"
        )
        resp = chat(req, db=test_db)
        assert resp.response == "Here is the final consolidated answer."

        metrics = test_db.query(RequestMetric).filter_by(session_id="session_capped_06").all()
        assert len(metrics) == 1
        assert metrics[0].path == "agent_capped"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Forced final call failure -> fallback message with error (turn not saved)
# ─────────────────────────────────────────────────────────────────────────────

def test_07_forced_final_call_fails():
    call_log = []

    def fake_generate_with_fallback(*args, **kwargs):
        call_log.append(kwargs)
        if len(call_log) <= 2:
            return LLMResponse(
                content="thinking",
                model="gemini-3.6-flash",
                provider="gemini",
                tool_calls=[ToolCall(id="c", name="get_job_status", arguments={})],
            )
        else:
            # 3rd call fails with non-cooldown error
            return LLMResponse(
                content="",
                model="gemini-3.6-flash",
                provider="gemini",
                finish_reason="error",
                error="Internal model error 500",
            )

    fake_factory = MagicMock()
    fake_factory.generate_with_fallback = fake_generate_with_fallback
    fake_registry = MagicMock()
    fake_tool = MagicMock()
    fake_tool.execute.return_value = ("out", ToolInvocation(name="t", arguments={}, ok=True, summary="ok"), [])
    fake_registry.get_tool.return_value = fake_tool
    fake_registry.get_all_tool_specs.return_value = []

    agent = ReActAgent(registry=fake_registry, max_iterations=2)
    agent.factory = fake_factory

    result = agent.run("lookup")
    assert result.error is not None
    assert "Exceeded maximum iteration cap" in result.error
    assert "lookup depth" in result.content


# ─────────────────────────────────────────────────────────────────────────────
# 8. ReActAgent() no arguments uses agent.max_iterations from config (2)
# ─────────────────────────────────────────────────────────────────────────────

def test_08_agent_default_max_iterations():
    agent = ReActAgent()
    assert agent.max_iterations == 2


# ─────────────────────────────────────────────────────────────────────────────
# 9. Gemini provider tool_choice config builder
# ─────────────────────────────────────────────────────────────────────────────

def test_09_gemini_provider_tool_choice_config():
    from llm.gemini_provider import GeminiProvider

    with patch("google.genai.Client"):
        provider = GeminiProvider(api_key="fake_key", model="gemini-3.6-flash")

        # tool_choice = "none"
        cfg_none = provider._config(tools=[], tool_choice="none")
        assert cfg_none.tool_config is not None
        assert cfg_none.tool_config.function_calling_config.mode == "NONE"

        # tool_choice = None
        cfg_default = provider._config(tools=[], tool_choice=None)
        assert getattr(cfg_default, "tool_config", None) is None


# ─────────────────────────────────────────────────────────────────────────────
# 10. cached_tokens extraction and summation
# ─────────────────────────────────────────────────────────────────────────────

def test_10_cached_tokens_extraction_and_summation():
    calls = 0

    def fake_generate_with_fallback(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LLMResponse(
                content="thought 1",
                model="gemini-3.6-flash",
                provider="gemini",
                cached_tokens=150,
                tool_calls=[ToolCall(id="c1", name="get_job_status", arguments={})],
            )
        else:
            return LLMResponse(
                content="final answer",
                model="gemini-3.6-flash",
                provider="gemini",
                cached_tokens=200,
                tool_calls=None,
            )

    fake_factory = MagicMock()
    fake_factory.generate_with_fallback = fake_generate_with_fallback
    fake_registry = MagicMock()
    fake_tool = MagicMock()
    fake_tool.execute.return_value = ("out", ToolInvocation(name="t", arguments={}, ok=True, summary="ok"), [])
    fake_registry.get_tool.return_value = fake_tool
    fake_registry.get_all_tool_specs.return_value = []

    agent = ReActAgent(registry=fake_registry, max_iterations=2)
    agent.factory = fake_factory

    result = agent.run("cached inquiry")
    assert result.cached_tokens == 350


# ─────────────────────────────────────────────────────────────────────────────
# 11. Prompt stability and react instructions rules
# ─────────────────────────────────────────────────────────────────────────────

def test_11_prompt_stability_and_rules():
    p1 = get_react_system_prompt()
    p2 = get_react_system_prompt()
    assert p1 == p2

    ri_text = (BACKEND / "prompts" / "react_instructions.md").read_text(encoding="utf-8")
    assert "SAME step" in ri_text
    assert "Before giving your final answer:" in ri_text
    assert "[Source: <title>, p. <page>]" in ri_text


# ─────────────────────────────────────────────────────────────────────────────
# 12. Parallel gatekeeper with blocked guardrail
# ─────────────────────────────────────────────────────────────────────────────

def test_12_parallel_gatekeeper_blocked(test_db):
    req = ChatRequest(message="I have severe head pain", session_id="session_par_blocked_12")

    mock_blocked_decision = GuardrailDecision(
        blocked=True,
        category="personal_medical",
        reason="Medical emergency",
        latency_ms=10.0,
        error=None,
        stage="input",
        model="gemini-3.5-flash-lite",
    )

    def slow_agent(*args, **kwargs):
        time.sleep(1.0)
        return ReActResult(
            content="Unsafe medical answer",
            provider_used="gemini",
            tool_calls=[ToolInvocation(name="get_job_status", arguments={"patient_id": "99"}, ok=True, summary="ok", record_ids=["99"])],
        )

    with patch("routers.chat.get_performance_config") as mock_perf, \
         patch("routers.chat.get_medical_guardrail") as mock_gr, \
         patch("routers.chat.orchestrator.run", side_effect=slow_agent):
        mock_perf.return_value.parallel_gatekeeper = True
        mock_gr.return_value.check_input.return_value = mock_blocked_decision

        t0 = time.perf_counter()
        resp = chat(req, db=test_db)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.8
        assert resp.guardrail == "medical_advice"

        from memory import get_memory_store
        history = get_memory_store().load("session_par_blocked_12")
        assert "Unsafe medical answer" not in [m.content for m in history]


# ─────────────────────────────────────────────────────────────────────────────
# 13. Parallel gatekeeper with allowed guardrail
# ─────────────────────────────────────────────────────────────────────────────

def test_13_parallel_gatekeeper_allowed(test_db):
    req = ChatRequest(message="What is the pipeline status?", session_id="session_par_allowed_13")

    mock_allowed_decision = GuardrailDecision(
        blocked=False,
        category="technical",
        reason="Technical question",
        latency_ms=10.0,
        error=None,
        stage="input",
        model="gemini-3.5-flash-lite",
    )
    mock_res = ReActResult(
        content="Pipeline is active.",
        provider_used="gemini",
        tokens_used=100,
        tool_calls=[],
        messages=[Message(role="user", content="req"), Message(role="assistant", content="Pipeline is active.")],
    )

    with patch("routers.chat.get_performance_config") as mock_perf, \
         patch("routers.chat.get_medical_guardrail") as mock_gr, \
         patch("routers.chat.orchestrator.run", return_value=mock_res):
        mock_perf.return_value.parallel_gatekeeper = True
        mock_gr.return_value.check_input.return_value = mock_allowed_decision

        resp = chat(req, db=test_db)
        assert resp.response == "Pipeline is active."

        metrics = test_db.query(RequestMetric).filter_by(session_id="session_par_allowed_13").all()
        assert len(metrics) == 1
        assert metrics[0].parallel is True


# ─────────────────────────────────────────────────────────────────────────────
# 14. All 6 return paths write exactly one RequestMetric row
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path_type", [
    "fast_path",
    "guardrail_block",
    "guardrail_route",
    "agent",
    "agent_capped",
    "agent_error",
])
def test_14_all_return_paths_write_metric(test_db, path_type):
    sess_id = f"sess_metric_{path_type}"

    if path_type == "fast_path":
        req = ChatRequest(message="Hi", session_id=sess_id)
        chat(req, db=test_db)

    elif path_type == "guardrail_block":
        req = ChatRequest(message="Medical headache", session_id=sess_id)
        with patch("routers.chat.get_medical_guardrail") as mock_gr:
            mock_gr.return_value.check_input.return_value = GuardrailDecision(
                blocked=True, category="personal_medical", reason="blocked", latency_ms=5.0, error=None, stage="input", model="gemini"
            )
            chat(req, db=test_db)

    elif path_type == "guardrail_route":
        req = ChatRequest(message="Out of scope query", session_id=sess_id)
        with patch("routers.chat.get_medical_guardrail") as mock_gr:
            mock_gr.return_value.check_input.return_value = GuardrailDecision(
                blocked=False, category="out_of_scope", reason="oos", latency_ms=5.0, error=None, stage="input", model="gemini"
            )
            chat(req, db=test_db)

    elif path_type == "agent":
        req = ChatRequest(message="Valid question", session_id=sess_id)
        res = ReActResult(content="Answer", provider_used="gemini", messages=[Message(role="user", content="q"), Message(role="assistant", content="a")])
        with patch("routers.chat.get_medical_guardrail") as mock_gr, \
             patch("routers.chat.orchestrator.run", return_value=res):
            mock_gr.return_value.check_input.return_value = GuardrailDecision(
                blocked=False, category="technical", reason="ok", latency_ms=5.0, error=None, stage="input", model="gemini"
            )
            chat(req, db=test_db)

    elif path_type == "agent_capped":
        req = ChatRequest(message="Complex question", session_id=sess_id)
        res = ReActResult(content="Consolidated answer", provider_used="gemini", capped=True, messages=[Message(role="user", content="q"), Message(role="assistant", content="a")])
        with patch("routers.chat.get_medical_guardrail") as mock_gr, \
             patch("routers.chat.orchestrator.run", return_value=res):
            mock_gr.return_value.check_input.return_value = GuardrailDecision(
                blocked=False, category="technical", reason="ok", latency_ms=5.0, error=None, stage="input", model="gemini"
            )
            chat(req, db=test_db)

    elif path_type == "agent_error":
        req = ChatRequest(message="Error question", session_id=sess_id)
        res = ReActResult(content="Error occurred", provider_used="gemini", error="Something broke")
        with patch("routers.chat.get_medical_guardrail") as mock_gr, \
             patch("routers.chat.orchestrator.run", return_value=res):
            mock_gr.return_value.check_input.return_value = GuardrailDecision(
                blocked=False, category="technical", reason="ok", latency_ms=5.0, error=None, stage="input", model="gemini"
            )
            chat(req, db=test_db)

    metrics = test_db.query(RequestMetric).filter_by(session_id=sess_id).all()
    assert len(metrics) == 1
    assert metrics[0].path == path_type


# ─────────────────────────────────────────────────────────────────────────────
# 15. Schema safety: copy feedback.db, run init_db(), execute requests
# ─────────────────────────────────────────────────────────────────────────────

def test_15_schema_safety():
    real_db_path = BACKEND / "data" / "database" / "feedback.db"
    if not real_db_path.exists():
        pytest.skip("feedback.db does not exist yet")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db_file = Path(tmp_dir) / "feedback.db"
        shutil.copy(real_db_path, tmp_db_file)

        tmp_engine = create_engine(f"sqlite:///{tmp_db_file}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=tmp_engine)
        TmpSession = sessionmaker(autocommit=False, autoflush=False, bind=tmp_engine)
        db_sess = TmpSession()

        try:
            # 1. Fast path request
            req_fp = ChatRequest(message="Hello", session_id="sess_tmp_fp")
            chat(req_fp, db=db_sess)

            # 2. Agent request
            res = ReActResult(content="Answer", provider_used="gemini", messages=[Message(role="user", content="q"), Message(role="assistant", content="a")])
            with patch("routers.chat.get_medical_guardrail") as mock_gr, \
                 patch("routers.chat.orchestrator.run", return_value=res):
                mock_gr.return_value.check_input.return_value = GuardrailDecision(
                    blocked=False, category="technical", reason="ok", latency_ms=5.0, error=None, stage="input", model="gemini"
                )
                req_ag = ChatRequest(message="Status of job", session_id="sess_tmp_ag")
                chat(req_ag, db=db_sess)

            # Verify rows exist in temp db
            rows = db_sess.query(RequestMetric).all()
            assert len(rows) >= 2
        finally:
            db_sess.close()
            tmp_engine.dispose()
