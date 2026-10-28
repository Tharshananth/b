"""
Job Status Tool for ReAct Agent.

Looks up the live processing status of neuroimaging jobs in MongoDB, strictly
filtered by the caller's server-enforced TenantScope.
"""
from typing import Any, Dict, List, Optional, Tuple
from pymongo.errors import PyMongoError

from agents.tools.base import BaseAgentTool
from services.job_status import ScopeViolation, TenantScope, get_user_job_status


class GetJobStatusTool(BaseAgentTool):
    """Tool for querying live patient and scan job processing statuses."""

    name = "get_job_status"
    description = (
        "Look up the live processing status of the caller's own neuroimaging jobs: "
        "which pipeline stage a scan reached, whether it succeeded, failed, or stalled, "
        "and overall status. Use this for questions about a specific scan, case, patient, "
        "or execution - e.g. 'is patient 2013 done', 'why did the upload fail'. "
        "Results are automatically scoped to the user's hospital/tenant."
    )
    parameters = {
        "type": "object",
        "properties": {
            "patient_id": {
                "type": "string",
                "description": "Patient identifier, e.g. '2013'. Omit to list recent jobs.",
            },
            "execution_id": {
                "type": "string",
                "description": "Execution identifier, e.g. 'Visit1_Dataentry'.",
            },
            "study_id": {
                "type": "string",
                "description": "Study identifier, e.g. '2300'.",
            },
            "trial_id": {
                "type": "integer",
                "description": "Trial identifier, e.g. 4584.",
            },
            "disorder": {
                "type": "string",
                "description": "Disorder category, e.g. 'Tumor'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of jobs to return. Defaults to 10.",
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
            msg = (
                "Identity is not available for this session, so job records cannot "
                "be looked up. Please inform the user that their job data cannot be "
                "accessed without an active hospital session."
            )
            return {"error": msg}, "no session scope", [], [], None

        allowed = {
            "patient_id", "execution_id", "study_id", "trial_id", "disorder", "limit"
        }
        clean = {k: v for k, v in args.items() if k in allowed and v is not None}

        try:
            rows = get_user_job_status(scope, **clean)
        except ScopeViolation:
            raise
        except PyMongoError as e:
            msg = (
                "The job database is currently unreachable. Inform the user you cannot "
                "check their job status right now - do not fabricate data."
            )
            return {"error": msg, "detail": str(e)}, "database unavailable", [], [], None

        summary = f"{len(rows)} job(s) found"
        if rows:
            summary += " (" + ", ".join(r.get("overall", "?") for r in rows[:5]) + ")"

        record_ids = [str(r.get("job_id")) for r in rows if r.get("job_id")]
        return rows, summary, record_ids, [], rows
