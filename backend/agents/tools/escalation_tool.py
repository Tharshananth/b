"""
Escalation Tool for ReAct Agent.

Formats support tickets and escalations to the BrainSightAI engineering team.
"""
from typing import Any, Dict, List, Optional, Tuple
from agents.tools.base import BaseAgentTool
from services.job_status import TenantScope


class EscalateToSupportTool(BaseAgentTool):
    """Tool to prepare an escalation to BrainSightAI technical support."""

    name = "escalate_to_support"
    description = (
        "Prepare an official escalation to BrainSightAI technical support "
        "(support@brainsightai.com) when an issue cannot be resolved through "
        "automated troubleshooting, when data corruption is identified, or when "
        "manual cluster intervention is required."
    )
    parameters = {
        "type": "object",
        "properties": {
            "issue_type": {
                "type": "string",
                "enum": ["pipeline_failure", "data_corruption", "cluster_stall", "account_or_access", "general"],
                "description": "Category of the issue requiring escalation.",
            },
            "summary": {
                "type": "string",
                "description": "Brief 1-sentence summary of the problem.",
            },
            "details": {
                "type": "string",
                "description": "Key diagnostic details, affected patient or scan IDs, and steps already attempted.",
            },
        },
        "required": ["issue_type", "summary", "details"],
    }

    def _run(
        self,
        args: Dict[str, Any],
        scope: Optional[TenantScope],
        context: Dict[str, Any],
    ) -> Tuple[Any, str, List[str], List[Dict[str, str]], Any]:
        issue_type = args.get("issue_type", "general")
        summary = args.get("summary", "")
        details = args.get("details", "")

        ticket_payload = {
            "status": "escalation_prepared",
            "support_email": "support@brainsightai.com",
            "hospital": scope.hospital if scope else "Unknown / Unspecified",
            "issue_type": issue_type,
            "summary": summary,
            "details": details,
            "recommended_next_step": "Instruct user to send email to support@brainsightai.com with these details or notify administrative contact.",
        }

        return ticket_payload, f"Escalation generated ({issue_type})", [], [], ticket_payload
