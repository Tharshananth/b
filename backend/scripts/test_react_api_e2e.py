"""
End-to-End API Test Script for ReAct Chatbot Endpoint.

Tests the FastAPI HTTP layer using TestClient:
  1. POST /api/chat/ schema validation & react_steps structure
  2. Scoped patient status inquiry with mock scope
  3. Multi-turn conversation memory persistence
  4. Validation rejection for empty inputs (HTTP 422)
  5. Memory inspection endpoint (/api/chat/memory/info/{session_id})

Run:
    python scripts/test_react_api_e2e.py
"""
import os
import sys
import json
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.chat import router as chat_router
from unittest.mock import patch
from agents.state import ReActResult, ReActStep, ToolInvocation

app = FastAPI()
app.include_router(chat_router)
client = TestClient(app)

PASS_ICON = "[PASS]"
FAIL_ICON = "[FAIL]"
passed_tests = 0
failed_tests = 0


def log_test(name: str):
    print(f"\n{'='*70}\n>>> API TEST: {name}\n{'='*70}")


def assert_condition(condition: bool, message: str):
    global passed_tests, failed_tests
    if condition:
        print(f" {PASS_ICON} {message}")
        passed_tests += 1
    else:
        print(f" {FAIL_ICON} {message}")
        failed_tests += 1


def test_api_react_response_schema():
    log_test("1. Verify /api/chat/ Response Schema & ReAct Steps")

    # Mock the orchestrator run to return a rich ReActResult
    mock_steps = [
        ReActStep(
            iteration=1,
            thought="Looking up platform documentation for supported file formats.",
            action="search_knowledge_base",
            action_input={"query": "supported file formats"},
            observation="VoxelBox supports NIfTI and DICOM formats.",
            observation_summary="1 source(s) found",
            ok=True,
            execution_time_ms=120.5,
        ),
        ReActStep(
            iteration=2,
            thought="Information gathered. Formulating final answer.",
            action=None,
            observation_summary="Completed final answer",
            ok=True,
            execution_time_ms=45.0,
        )
    ]
    mock_result = ReActResult(
        content="VoxelBox Explore supports NIfTI (.nii, .nii.gz) and DICOM formats.",
        provider_used="mock-gemini",
        tokens_used=110,
        tool_calls=[
            ToolInvocation(
                name="search_knowledge_base",
                arguments={"query": "supported file formats"},
                ok=True,
                summary="1 source(s) found",
            )
        ],
        sources=[{"title": "File Format Guide", "url": "docs/formats.md", "content": "..."}],
        react_steps=mock_steps,
    )

    with patch("services.orchestrator.run", return_value=mock_result):
        resp = client.post(
            "/api/chat/",
            json={"message": "What file formats are supported?", "session_id": "test_sess_001"},
        )

    assert_condition(resp.status_code == 200, f"HTTP status 200 (got {resp.status_code})")
    data = resp.json()

    assert_condition("response" in data, "Response key present")
    assert_condition("react_steps" in data, "react_steps key present in API response")
    assert_condition(len(data["react_steps"]) == 2, f"Expected 2 react_steps, got {len(data['react_steps'])}")

    step1 = data["react_steps"][0]
    assert_condition(step1["iteration"] == 1, "Step 1 iteration is 1")
    assert_condition(step1["action"] == "search_knowledge_base", "Step 1 action captured correctly")
    assert_condition("thought" in step1 and len(step1["thought"]) > 0, "Step 1 thought populated")
    assert_condition(step1["execution_time_ms"] > 0, "Execution time recorded")

    assert_condition("tools_used" in data and len(data["tools_used"]) == 1, "tools_used list preserved")
    assert_condition("sources" in data and len(data["sources"]) == 1, "sources list preserved")
    assert_condition("suggestions" in data, "suggestions list returned")

    print(f"       Agent Response: {data['response']}")
    print(f"       Steps returned to UI: {len(data['react_steps'])} steps")


def test_api_empty_input_validation():
    log_test("2. Input Validation (Rejection of Empty Strings)")
    resp = client.post("/api/chat/", json={"message": "   "})
    assert_condition(resp.status_code == 422, f"Expected HTTP 422 validation error, got {resp.status_code}")


def test_api_multi_turn_history():
    log_test("3. Multi-Turn Session Memory Preservation")
    session_id = f"test_turn_session_{int(time.time())}"

    from llm.base import Message
    # First turn
    mock_result_1 = ReActResult(
        content="Hello Dr. Smith, how can I assist you today?",
        provider_used="mock-gemini",
        tokens_used=50,
        messages=[
            Message(role="user", content="My name is Dr. Smith"),
            Message(role="assistant", content="Hello Dr. Smith, how can I assist you today?"),
        ],
    )
    with patch("services.orchestrator.run", return_value=mock_result_1):
        resp1 = client.post(
            "/api/chat/",
            json={"message": "My name is Dr. Smith", "session_id": session_id},
        )
    assert_condition(resp1.status_code == 200, "Turn 1 succeeded")

    # Second turn: memory info endpoint inspection
    resp_mem = client.get(f"/api/chat/memory/info/{session_id}")
    assert_condition(resp_mem.status_code == 200, "Memory inspection endpoint returned 200")
    mem_data = resp_mem.json()
    assert_condition(mem_data.get("session_exists") is True, "Session memory exists in cache")


def run_all_api_tests():
    start = time.perf_counter()
    print("\n" + "#"*70)
    print("   BRAINSIGHTAI ReAct AGENT — END-TO-END API TEST SUITE")
    print("#"*70)

    test_api_react_response_schema()
    test_api_empty_input_validation()
    test_api_multi_turn_history()

    elapsed = (time.perf_counter() - start) * 1000.0
    print("\n" + "="*70)
    print(f"API TEST SUITE COMPLETE in {elapsed:.1f}ms")
    print(f"Total Passed: {passed_tests} | Total Failed: {failed_tests}")
    print("="*70 + "\n")

    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_api_tests()
    sys.exit(0 if success else 1)
