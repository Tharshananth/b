"""
Tests for Gemini model-level fallback, health tracking, and gatekeeper timeout fixes.

All tests run offline (no external network requests).
"""
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure backend directory is in sys.path
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

# Ensure real packages are loaded (undo any stubs left by other test files)
import importlib.util
_factory_spec = importlib.util.spec_from_file_location(
    "llm.factory", _backend / "llm" / "factory.py"
)
_real_factory_mod = importlib.util.module_from_spec(_factory_spec)
_factory_spec.loader.exec_module(_real_factory_mod)
LLMFactory = _real_factory_mod.LLMFactory
sys.modules["llm.factory"] = _real_factory_mod

import pytest

from config import get_config, MedicalGuardrailConfig
from llm.base import Message, ToolCall, LLMResponse
from llm.gemini_provider import GeminiProvider
from llm import model_health
from agents.react_agent import ReActAgent
from agents.state import ToolInvocation


@pytest.fixture(autouse=True)
def reset_model_health_state():
    """Reset health tracker between tests."""
    model_health.reset_health()
    yield
    model_health.reset_health()


# ── Test 1: First model 429 with retryDelay -> second model answers ────────────

def test_01_first_model_429_retry_delay_second_answers():
    """1. The first model returns 429 with retryDelay '120s' -> second answers;
    response.model is second; first is cooling down for ~120s."""
    model_A = "gemini-3.6-flash"
    model_B = "gemini-3.5-flash"

    provider = GeminiProvider(
        api_key="fake-key",
        model=model_A,
        fallback_models=[model_B],
    )
    provider.client = MagicMock()

    # Model A raises 429 with retryDelay "120s", Model B returns content
    exc_429 = Exception("429 RESOURCE_EXHAUSTED: quota exceeded. {'retryDelay': '120s'}")
    resp_B = MagicMock()
    resp_B.candidates = [MagicMock()]
    resp_B.candidates[0].content.parts = [MagicMock(text="Hello from model B", function_call=None)]
    resp_B.usage_metadata.total_token_count = 42

    def side_effect(model, contents, config):
        if model == model_A:
            raise exc_429
        elif model == model_B:
            return resp_B
        raise ValueError(f"Unexpected model {model}")

    provider.client.models.generate_content.side_effect = side_effect

    factory = LLMFactory()
    factory._providers = {"gemini": provider}

    messages = [Message(role="user", content="Hi")]
    response = factory.generate_with_fallback(messages)

    assert response.finish_reason == "stop"
    assert response.content == "Hello from model B"
    assert response.model == model_B
    assert not model_health.is_available(model_A)
    assert model_health.is_available(model_B)

    # First model cooling down for ~120 s
    status_A = model_health.status().get(model_A)
    assert status_A is not None
    assert status_A["status"] == "cooling_down"
    assert 115 <= status_A["seconds_remaining"] <= 125


# ── Test 2: Next call skips the cooling model entirely ───────────────────────

def test_02_next_call_skips_cooling_model():
    """2. The next call skips the cooling model entirely (it is not called)."""
    model_A = "gemini-3.6-flash"
    model_B = "gemini-3.5-flash"

    # Put model A in cooldown
    model_health.mark_unavailable(model_A, 120, "429 RESOURCE_EXHAUSTED")

    provider = GeminiProvider(
        api_key="fake-key",
        model=model_A,
        fallback_models=[model_B],
    )
    provider.client = MagicMock()

    resp_B = MagicMock()
    resp_B.candidates = [MagicMock()]
    resp_B.candidates[0].content.parts = [MagicMock(text="Model B direct", function_call=None)]
    resp_B.usage_metadata.total_token_count = 20

    provider.client.models.generate_content.return_value = resp_B

    factory = LLMFactory()
    factory._providers = {"gemini": provider}

    messages = [Message(role="user", content="Next message")]
    response = factory.generate_with_fallback(messages)

    assert response.model == model_B
    # Assert model_A was never requested from the client
    for call_item in provider.client.models.generate_content.call_args_list:
        assert call_item.kwargs.get("model") != model_A


# ── Test 3: 400 error is not marked and not retried ──────────────────────────

def test_03_400_error_not_marked_and_not_retried():
    """3. A 400 error is NOT marked and is not retried on other models."""
    model_A = "gemini-3.6-flash"
    model_B = "gemini-3.5-flash"

    provider = GeminiProvider(
        api_key="fake-key",
        model=model_A,
        fallback_models=[model_B],
    )
    provider.client = MagicMock()
    provider.client.models.generate_content.side_effect = Exception("400 INVALID_ARGUMENT: Bad request syntax")

    factory = LLMFactory()
    factory._providers = {"gemini": provider}

    messages = [Message(role="user", content="Trigger 400")]
    response = factory.generate_with_fallback(messages)

    assert response.finish_reason == "error"
    # Model A is NOT marked unavailable
    assert model_health.is_available(model_A)
    # Model B was never called
    assert provider.client.models.generate_content.call_count == 1
    call_kwargs = provider.client.models.generate_content.call_args.kwargs
    assert call_kwargs.get("model") == model_A


# ── Test 4: 404 -> 3600s cooldown; 503 -> 30s ───────────────────────────────

def test_04_cooldown_rules_404_and_503():
    """4. A 404 -> 3600 s cooldown; a 503 -> 30 s."""
    cooldown_404 = model_health.get_cooldown_for_exception(Exception("404 NOT_FOUND: model not found"))
    assert cooldown_404 is not None
    assert cooldown_404[0] == 3600

    cooldown_503 = model_health.get_cooldown_for_exception(Exception("503 UNAVAILABLE: high demand"))
    assert cooldown_503 is not None
    assert cooldown_503[0] == 30

    cooldown_timeout = model_health.get_cooldown_for_exception(TimeoutError("Request timed out"))
    assert cooldown_timeout is not None
    assert cooldown_timeout[0] == 30


# ── Test 5: All models cooling down -> still tries them ─────────────────────

def test_05_all_models_cooling_down_still_tries_them():
    """5. All models cooling down -> still tries them (no instant refusal)."""
    model_A = "gemini-3.6-flash"
    model_B = "gemini-3.5-flash"

    # Mark both models as cooling down
    model_health.mark_unavailable(model_A, 60, "429")
    model_health.mark_unavailable(model_B, 60, "429")

    provider = GeminiProvider(
        api_key="fake-key",
        model=model_A,
        fallback_models=[model_B],
    )
    provider.client = MagicMock()

    # Model A still fails, but Model B unexpectedly succeeds!
    resp_B = MagicMock()
    resp_B.candidates = [MagicMock()]
    resp_B.candidates[0].content.parts = [MagicMock(text="Surprise recovery", function_call=None)]
    resp_B.usage_metadata.total_token_count = 10

    def side_effect(model, contents, config):
        if model == model_A:
            raise Exception("429 RESOURCE_EXHAUSTED")
        return resp_B

    provider.client.models.generate_content.side_effect = side_effect

    factory = LLMFactory()
    factory._providers = {"gemini": provider}

    # candidate_models() should NOT be empty even though all are in cooldown
    cands = provider.candidate_models()
    assert len(cands) == 2
    assert cands == [model_A, model_B]

    messages = [Message(role="user", content="Try anyway")]
    response = factory.generate_with_fallback(messages)

    assert response.finish_reason == "stop"
    assert response.content == "Surprise recovery"
    assert response.model == model_B


# ── Test 6: Agent round 1 on model A; round 2 also uses model A ─────────────

def test_06_agent_locks_chosen_model_in_round_2():
    """6. Agent: round 1 on model A; round 2 also uses model A (model= passed)."""
    model_A = "gemini-3.6-flash"

    factory = MagicMock()

    # Round 1: Model A returns a tool call
    tc = ToolCall(id="tc_1", name="dummy_tool", arguments={"x": 1})
    resp_1 = LLMResponse(
        content="",
        model=model_A,
        provider="gemini",
        finish_reason="tool_calls",
        tool_calls=[tc],
    )

    # Round 2: Model A returns final answer
    resp_2 = LLMResponse(
        content="Final answer from A",
        model=model_A,
        provider="gemini",
        finish_reason="stop",
    )

    factory.generate_with_fallback.side_effect = [resp_1, resp_2]

    registry = MagicMock()
    mock_tool = MagicMock()
    mock_tool.execute.return_value = (
        '{"status": "ok"}',
        ToolInvocation(name="dummy_tool", arguments={"x": 1}, ok=True, summary="done"),
        [],
    )
    registry.get_all_tool_specs.return_value = []
    registry.get_tool.return_value = mock_tool

    agent = ReActAgent(registry=registry)
    agent.factory = factory

    result = agent.run(user_message="Do dummy task")

    assert result.content == "Final answer from A"
    assert result.model_used == model_A
    assert factory.generate_with_fallback.call_count == 2

    # Verify round 1 had model=None (or not passed)
    r1_kwargs = factory.generate_with_fallback.call_args_list[0].kwargs
    assert r1_kwargs.get("model") is None

    # Verify round 2 had model=model_A passed
    r2_kwargs = factory.generate_with_fallback.call_args_list[1].kwargs
    assert r2_kwargs.get("model") == model_A


# ── Test 7: Model A 429 in round 2 -> restarts on model B from original messages ──

def test_07_agent_model_a_429_round_2_restarts_on_model_b():
    """7. Agent: model A fails with 429 in round 2 -> the question restarts on model B
    from the original messages; model B never receives model A's tool calls; the final
    model_used is B."""
    model_A = "gemini-3.6-flash"
    model_B = "gemini-3.5-flash"

    gemini_provider = MagicMock()
    gemini_provider.candidate_models.return_value = [model_A, model_B]

    factory = MagicMock()
    factory.get_provider.return_value = gemini_provider

    # Call sequences:
    # 1. Round 1 (model=None) -> returns Model A tool call
    tc_A = ToolCall(id="tc_A", name="tool_for_A", arguments={"from": "A"}, provider_data={"thought_signature": "sigA"})
    resp_1 = LLMResponse(
        content="",
        model=model_A,
        provider="gemini",
        finish_reason="tool_calls",
        tool_calls=[tc_A],
    )

    # 2. Round 2 (model=model_A) -> Model A fails with 429
    resp_2 = LLMResponse(
        content="I apologize...",
        model=model_A,
        provider="gemini",
        finish_reason="error",
        error="429 RESOURCE_EXHAUSTED",
    )

    # 3. Restart! Round 1 on Model B -> returns Model B tool call
    tc_B = ToolCall(id="tc_B", name="tool_for_B", arguments={"from": "B"}, provider_data={"thought_signature": "sigB"})
    resp_3 = LLMResponse(
        content="",
        model=model_B,
        provider="gemini",
        finish_reason="tool_calls",
        tool_calls=[tc_B],
    )

    # 4. Restart! Round 2 on Model B -> Model B final answer
    resp_4 = LLMResponse(
        content="Final answer from B",
        model=model_B,
        provider="gemini",
        finish_reason="stop",
    )

    recorded_messages_per_call = []

    def mock_generate(messages, **kwargs):
        recorded_messages_per_call.append((kwargs.get("model"), list(messages)))
        if len(recorded_messages_per_call) == 1:
            return resp_1
        elif len(recorded_messages_per_call) == 2:
            model_health.mark_unavailable(model_A, 120, "429")
            return resp_2
        elif len(recorded_messages_per_call) == 3:
            return resp_3
        else:
            return resp_4

    factory.generate_with_fallback.side_effect = mock_generate

    registry = MagicMock()
    mock_tool = MagicMock()
    mock_tool.execute.return_value = (
        '{"status": "ok"}',
        ToolInvocation(name="test_tool", arguments={}, ok=True, summary="done"),
        [],
    )
    registry.get_all_tool_specs.return_value = []
    registry.get_tool.return_value = mock_tool

    agent = ReActAgent(registry=registry)
    agent.factory = factory

    result = agent.run(user_message="Original inquiry")

    assert result.content == "Final answer from B"
    assert result.model_used == model_B
    assert len(recorded_messages_per_call) == 4

    # Call 3 (restart on Model B) must have had ONLY the user turn, NOT Model A's tool calls
    call_3_messages = recorded_messages_per_call[2][1]
    assert len(call_3_messages) == 1
    assert call_3_messages[0].role == "user"
    assert call_3_messages[0].content == "Original inquiry"

    # Model B never received Model A's tool calls or signatures
    for mod, msgs in recorded_messages_per_call[2:]:
        for m in msgs:
            if m.tool_calls:
                for t in m.tool_calls:
                    assert t.name != "tool_for_A"
                    if t.provider_data:
                        assert t.provider_data.get("thought_signature") != "sigA"


# ── Test 8: Gatekeeper lite returns 429 -> flash answers, logged in DB ───────

def test_08_gatekeeper_lite_429_flash_answers_and_logged(tmp_path):
    """8. Gatekeeper: lite returns 429 -> flash answers; decision.model == 'gemini-3.5-flash';
    logged in guardrail_events."""
    from guardrails.medical import MedicalGuardrail
    from database.models import GuardrailEvent
    from routers.chat import _log_guardrail_event
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database.models import Base

    db_path = tmp_path / "test_gr.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    cfg = MagicMock()
    cfg.model = "gemini-3.5-flash-lite"
    cfg.fallback_models = ["gemini-3.5-flash"]
    cfg.timeout_s = 10.0
    cfg.history_turns = 2
    cfg.on_error = "allow"

    gr = MedicalGuardrail(cfg=cfg)
    mock_client = MagicMock()
    gr._client = mock_client

    from guardrails.medical import Category

    resp_flash = MagicMock()
    resp_flash.parsed = MagicMock(category=Category.technical, reason="valid tech query")

    def side_effect(model, contents, config):
        if model == "gemini-3.5-flash-lite":
            raise Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
        elif model == "gemini-3.5-flash":
            return resp_flash
        raise ValueError(f"Unknown {model}")

    mock_client.models.generate_content.side_effect = side_effect

    decision = gr.check_input("How does MRI processing work in VoxelBox?", [])

    assert not decision.blocked
    assert decision.category == "technical"
    assert decision.model == "gemini-3.5-flash"

    # Log in guardrail_events
    _log_guardrail_event(
        db,
        session_id="s123",
        user_id="u123",
        message_id="m123",
        decision=decision,
        model=decision.model,
    )

    event = db.query(GuardrailEvent).filter_by(message_id="m123").first()
    assert event is not None
    assert event.model == "gemini-3.5-flash"
    assert event.decision == "allowed"
    assert event.stage == "input"


# ── Test 9: Gatekeeper timeout 10000ms & config warning on < 10 ──────────────

def test_09_gatekeeper_timeout_10000ms_and_config_validation(caplog):
    """9. Gatekeeper timeout is sent as 10 000 ms; a config value of 3 is raised to 10 with a warning."""
    # Test config validator raises 3 to 10
    with caplog.at_level(logging.WARNING):
        cfg = MedicalGuardrailConfig(timeout_s=3.0)
    assert cfg.timeout_s == 10.0
    assert "below Google minimum 10.0s; adjusting to 10.0s" in caplog.text

    # Test MedicalGuardrail._do_check_input sends 10000 ms
    from guardrails.medical import MedicalGuardrail

    gr = MedicalGuardrail(cfg=cfg)
    mock_client = MagicMock()
    gr._client = mock_client

    resp = MagicMock()
    resp.parsed = MagicMock(category=MagicMock(value="greeting"), reason="hello")
    mock_client.models.generate_content.return_value = resp

    gr.check_input("hello", [])

    call_config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert call_config.http_options.timeout == 10000


# ── Test 10: /health includes llm_models with cooldown info ──────────────────

def test_10_health_endpoint_includes_llm_models():
    """10. /health includes llm_models with cooldown info."""
    from fastapi.testclient import TestClient
    from main import app

    model_health.mark_unavailable("gemini-3.6-flash", 120, "429 RESOURCE_EXHAUSTED")
    model_health.register_model("gemini-3.5-flash")

    client = TestClient(app)
    res = client.get("/api/health/")
    assert res.status_code == 200
    data = res.json()

    assert "llm_models" in data
    llm_models = data["llm_models"]
    assert "gemini-3.6-flash" in llm_models
    assert llm_models["gemini-3.6-flash"]["status"] == "cooling_down"
    assert llm_models["gemini-3.6-flash"]["seconds_remaining"] > 0
    assert "gemini-3.5-flash" in llm_models
    assert llm_models["gemini-3.5-flash"]["status"] == "available"


# ── Test 11: fallback_order contains only gemini; OpenAI/Claude never called ──

def test_11_fallback_order_only_gemini():
    """11. fallback_order contains only gemini; OpenAI/Claude are never called."""
    cfg = get_config()
    assert cfg.llm.fallback_order == ["gemini"]

    # Test with factory: when gemini fails completely, OpenAI and Claude are never called
    factory = LLMFactory()
    gemini_mock = MagicMock()
    gemini_mock.candidate_models.return_value = ["gemini-3.6-flash"]
    gemini_mock.generate_response.side_effect = Exception("429 RESOURCE_EXHAUSTED")
    gemini_mock.supports_tools = True

    openai_mock = MagicMock()
    claude_mock = MagicMock()

    factory._providers = {
        "gemini": gemini_mock,
        "openai": openai_mock,
        "claude": claude_mock,
    }

    resp = factory.generate_with_fallback([Message(role="user", content="hello")])

    assert resp.finish_reason == "error"
    assert openai_mock.generate_response.call_count == 0
    assert claude_mock.generate_response.call_count == 0
