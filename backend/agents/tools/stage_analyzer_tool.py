"""
Job Stage Diagnostics Tool for ReAct Agent.

Performs deep analysis of neuroimaging pipeline execution stages, anomaly detection,
stall classification, and failure root-cause explanation using execution_summary.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pymongo.errors import PyMongoError

from agents.tools.base import BaseAgentTool
from services.execution_summary import (
    _find_problems,
    _flatten,
    explain_failure,
    classify,
    FAILED,
    RUNNING,
)
from services.job_status import ScopeViolation, TenantScope, get_job_status


class DiagnoseJobStagesTool(BaseAgentTool):
    """Tool for in-depth pipeline stage diagnostics and failure root-cause analysis."""

    name = "diagnose_job_stages"
    description = (
        "Diagnose pipeline issues for a specific job or patient. Analyzes every "
        "stage (dataUpload, dicomToNifti, preprocess, inference), detects stalled "
        "tasks, checks timing anomalies, and extracts the exact root cause and next steps "
        "for failed scans. Use this when a job has failed, stalled, or needs deep troubleshooting."
    )
    parameters = {
        "type": "object",
        "properties": {
            "patient_id": {
                "type": "string",
                "description": "Patient identifier to diagnose.",
            },
            "execution_id": {
                "type": "string",
                "description": "Specific execution identifier to diagnose.",
            },
        },
        "required": [],
    }

    def _run(
        self,
        args: Dict[str, Any],
        scope: Optional[TenantScope],
        context: Dict[str, Any],
    ) -> Tuple[Any, str, List[str], List[Dict[str, str]], Any]:
        if scope is None:
            return (
                {"error": "Session identity is required to diagnose pipeline stages."},
                "no session scope",
                [],
                [],
                None,
            )

        patient_id = args.get("patient_id")
        execution_id = args.get("execution_id")

        if not patient_id and not execution_id:
            return (
                {"error": "Please provide either patient_id or execution_id to diagnose."},
                "missing identifier",
                [],
                [],
                None,
            )

        try:
            docs = get_job_status(
                scope,
                patient_id=patient_id,
                execution_id=execution_id,
                limit=3,
            )
        except ScopeViolation:
            raise
        except PyMongoError as e:
            return (
                {"error": f"Failed to retrieve execution records for diagnostics: {e}"},
                "db error",
                [],
                [],
                None,
            )

        if not docs:
            return (
                {"diagnostic": f"No job records found matching patient_id={patient_id}, execution_id={execution_id}."},
                "no jobs found",
                [],
                [],
                None,
            )

        diagnostics = []
        record_ids = []

        now = datetime.now()
        for doc in docs:
            job_id = str(doc.get("_id", "unknown"))
            record_ids.append(job_id)

            status = doc.get("status") or {}
            timespan = doc.get("timespan") or {}
            docker_arr = doc.get("dockerArray") or []

            stages = _flatten(status, timespan)
            problems = _find_problems(stages, docker_arr, now)

            failed_explanations = []
            for st in stages:
                if st.get("outcome") == FAILED:
                    expl = explain_failure(str(st.get("state")), str(st.get("stage")))
                    failed_explanations.append({
                        "stage": st.get("stage"),
                        "task": st.get("task"),
                        "raw_status": str(st.get("state")),
                        "cause": expl.get("cause"),
                        "action": expl.get("action"),
                        "terminal": expl.get("terminal"),
                    })

            diagnostics.append({
                "job_id": job_id,
                "patient_id": doc.get("patientID"),
                "execution_id": doc.get("executionID"),
                "study_id": doc.get("studyID"),
                "disorder": doc.get("disorder"),
                "detected_problems": problems,
                "failed_stage_details": failed_explanations,
                "stages_breakdown": [
                    {
                        "stage": s["stage"],
                        "task": s["task"],
                        "outcome": s["outcome"],
                        "duration_seconds": s["duration_seconds"],
                    }
                    for s in stages
                    if not s["is_marker"]
                ],
            })

        summary = f"Diagnosed {len(diagnostics)} job(s); {sum(len(d['detected_problems']) for d in diagnostics)} anomaly/anomalies detected"
        return diagnostics, summary, record_ids, [], diagnostics
