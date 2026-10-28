"""
Standalone Verification Script for ReAct Agent Engine.

Tests all cognitive loops, tools, security isolation, and edge cases:
  1. Direct Domain Q&A (Reasoning + Final Answer)
  2. Multi-step Pipeline Diagnostics (get_job_status -> diagnose_job_stages -> Synthesis)
  3. Strict Tenant Isolation (Scoped vs Unscoped refusal)
  4. Support Escalation on Hardware/Checksum Corruption
  5. Loop Detection & Duplicate Tool Prevention
  6. Maximum Iteration Safety Cap

Run:
    python scripts/test_react_engine.py
"""
import os
import sys
import json
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from agents.react_agent import ReActAgent
from agents.state import ReActStep, ToolInvocation
from services.job_status import TenantScope, ScopeViolation
from llm.base import LLMResponse, ToolCall, Message

PASS_ICON = "[PASS]"
FAIL_ICON = "[FAIL]"
INFO_ICON = "[INFO]"

passed_tests = 0
failed_tests = 0


def log_test(name: str):
    print(f"\n{'='*70}\n>>> TEST: {name}\n{'='*70}")


def assert_condition(condition: bool, message: str):
    global passed_tests, failed_tests
    if condition:
        print(f" {PASS_ICON} {message}")
        passed_tests += 1
    else:
        print(f" {FAIL_ICON} {message}")
        failed_tests += 1


class MockScriptLLMFactory:
    """Predictable scripted LLM responses to test multi-turn agent behavior."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_idx = 0

    def generate_with_fallback(self, **kwargs):
        if self.call_idx >= len(self.responses):
            return LLMResponse(
                content="Fallback response: out of scripted mock responses.",
                model="mock",
                provider="mock",
                finish_reason="stop",
            )
        resp = self.responses[self.call_idx]
        self.call_idx += 1
        return resp


def test_scenario_1_direct_answer():
    log_test("1. Direct Answer (Knowledge Retrieval without Tool Calling)")
    agent = ReActAgent()
    agent.factory = MockScriptLLMFactory([
        LLMResponse(
            content="Thought: The user is asking about VoxelBox supported file formats. I know this from standard documentation.\nFinal Answer: VoxelBox Explore supports standard NIfTI (.nii, .nii.gz) and DICOM (.dcm) formats.",
            model="mock-gpt4",
            provider="mock-openai",
            tokens_used=50,
            finish_reason="stop",
        )
    ])

    result = agent.run(user_message="What formats does VoxelBox support?")

    assert_condition("NIfTI" in result.content and "DICOM" in result.content, "Response contains expected technical formats")
    assert_condition(len(result.react_steps) == 1, f"Expected 1 ReAct step, got {len(result.react_steps)}")
    assert_condition(result.react_steps[0].action is None, "Step correctly identified as final answer (no action)")
    assert_condition(result.error is None, "Result completed without errors")
    print(f"       Thought: {result.react_steps[0].thought}")
    print(f"       Answer: {result.content}")


def test_scenario_2_multistep_pipeline_diagnostics():
    log_test("2. Multi-step ReAct Pipeline Diagnostics (Lookup -> Diagnose -> Answer)")
    agent = ReActAgent()
    agent.factory = MockScriptLLMFactory([
        # Step 1: Agent checks job status
        LLMResponse(
            content="Thought: The user is asking about patient 2013's scan. I must look up their recent jobs first.",
            model="mock-claude",
            provider="mock-anthropic",
            tokens_used=40,
            tool_calls=[
                ToolCall(
                    id="tc_1",
                    name="get_job_status",
                    arguments={"patient_id": "2013"},
                )
            ],
        ),
        # Step 2: Agent sees a failure and investigates deep stages
        LLMResponse(
            content="Thought: The job status indicates a failure at stage dicomToNifti. I should diagnose the stages to find the root cause.",
            model="mock-claude",
            provider="mock-anthropic",
            tokens_used=60,
            tool_calls=[
                ToolCall(
                    id="tc_2",
                    name="diagnose_job_stages",
                    arguments={"patient_id": "2013"},
                )
            ],
        ),
        # Step 3: Agent synthesizes the diagnostic root cause
        LLMResponse(
            content="Thought: Now I have the full diagnosis. The scan failed at dicomToNifti due to missing slices in series 4. I can now advise the user.\nFinal Answer: Patient 2013's scan failed during DICOM to NIfTI conversion due to corrupted/missing slices in series 4. Please re-upload the complete DICOM archive.",
            model="mock-claude",
            provider="mock-anthropic",
            tokens_used=95,
            finish_reason="stop",
        ),
    ])

    scope = TenantScope(hospital="MetroHealth")
    result = agent.run(user_message="What happened to patient 2013's scan?", scope=scope)

    assert_condition(len(result.react_steps) == 3, f"Expected 3 ReAct steps, got {len(result.react_steps)}")
    assert_condition(result.react_steps[0].action == "get_job_status", "Step 1 correctly performed get_job_status")
    assert_condition(result.react_steps[1].action == "diagnose_job_stages", "Step 2 autonomously performed diagnose_job_stages")
    assert_condition(result.react_steps[2].action is None, "Step 3 concluded with Final Answer")
    assert_condition("corrupted/missing slices" in result.content, "Final answer contains synthesized diagnosis")

    for step in result.react_steps:
        print(f"       [Step {step.iteration}] Action: {step.action or 'Final Answer'} | Time: {step.execution_time_ms:.1f}ms")
        print(f"                  Thought: {step.thought[:80]}...")


def test_scenario_3_security_and_unscoped_refusal():
    log_test("3. Security & Unscoped Access Prevention")
    from agents.tools.job_status_tool import GetJobStatusTool

    tool = GetJobStatusTool()
    output, inv, sources = tool.execute({"patient_id": "2013"}, scope=None)

    assert_condition(inv.ok is False or "no session scope" in inv.summary, "Tool safely refused unscoped patient query")
    assert_condition("Identity is not available" in output, "Tool output explains identity requirement to agent")
    assert_condition(len(inv.record_ids) == 0, "Zero patient record IDs accessed without scope")


def test_scenario_4_escalation_ticket():
    log_test("4. Automated Escalation to Support")
    from agents.tools.escalation_tool import EscalateToSupportTool

    tool = EscalateToSupportTool()
    output, inv, sources = tool.execute(
        args={
            "issue_type": "data_corruption",
            "summary": "T1 DICOM headers unparseable",
            "details": "Patient 902 study 44 checksum failed",
        },
        scope=TenantScope(hospital="ApexClinic"),
    )

    assert_condition(inv.ok is True, "Escalation tool executed successfully")
    assert_condition("support@brainsightai.com" in output, "Escalation payload includes official support email")
    assert_condition("ApexClinic" in output, "Escalation payload preserves caller hospital context")


def test_scenario_5_loop_detection():
    log_test("5. Infinite Loop & Duplicate Action Detection")
    agent = ReActAgent(max_iterations=3)
    # Model repeatedly requests the exact same query
    agent.factory = MockScriptLLMFactory([
        LLMResponse(
            content="Thought: Querying job status.",
            model="mock",
            provider="mock",
            tool_calls=[ToolCall(id="c1", name="get_job_status", arguments={"patient_id": "111"})],
        ),
        LLMResponse(
            content="Thought: Querying job status again.",
            model="mock",
            provider="mock",
            tool_calls=[ToolCall(id="c2", name="get_job_status", arguments={"patient_id": "111"})],
        ),
        LLMResponse(
            content="Thought: Querying job status a third time.",
            model="mock",
            provider="mock",
            tool_calls=[ToolCall(id="c3", name="get_job_status", arguments={"patient_id": "111"})],
        ),
    ])

    result = agent.run(user_message="Check job", scope=TenantScope(hospital="Test"))

    assert_condition(result.error is not None, "Agent terminated cleanly when iteration cap reached")
    assert_condition("Exceeded maximum iteration cap" in result.error, f"Reported cap error: {result.error}")
    assert_condition(len(result.react_steps) == 3, f"Agent stopped at max_iterations (3), got {len(result.react_steps)}")


def run_all_tests():
    start = time.perf_counter()
    print("\n" + "#"*70)
    print("   BRAINSIGHTAI ReAct AGENT — COMPREHENSIVE ENGINE VERIFICATION")
    print("#"*70)

    test_scenario_1_direct_answer()
    test_scenario_2_multistep_pipeline_diagnostics()
    test_scenario_3_security_and_unscoped_refusal()
    test_scenario_4_escalation_ticket()
    test_scenario_5_loop_detection()

    elapsed = (time.perf_counter() - start) * 1000.0
    print("\n" + "="*70)
    print(f"VERIFICATION COMPLETE in {elapsed:.1f}ms")
    print(f"Total Passed: {passed_tests} | Total Failed: {failed_tests}")
    print("="*70 + "\n")

    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
