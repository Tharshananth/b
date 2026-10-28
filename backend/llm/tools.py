"""
Tool definitions exposed to the model.

One plain JSON Schema per tool, translated per provider at the call site.
OpenAI and Anthropic take JSON Schema directly; google-genai accepts it through
FunctionDeclaration.parameters_json_schema.

The scope fields - hospital above all - are deliberately absent from every
schema here. Scope comes from the authenticated session and is injected
server-side in services.job_status, so a model cannot name it, cannot widen it,
and cannot be argued into changing it. Adding a tenant field to one of these
schemas would undo that.
"""
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may request, described once for all providers."""
    name: str
    description: str
    parameters: Dict[str, Any]   # JSON Schema, object type


GET_JOB_STATUS = ToolSpec(
    name="get_job_status",
    description=(
        "Look up the live processing status of the caller's own neuroimaging "
        "jobs: which pipeline stage a scan reached, whether it succeeded, "
        "failed or stalled, why it failed, and what to do next. Use this for "
        "any question about a specific scan, case, patient or run - for "
        "example 'is my scan done', 'why did the upload fail', 'what stage is "
        "patient 2013 at'. Do not use it for questions about how the platform "
        "works in general; use search_knowledge_base for those. Results are "
        "automatically limited to data the caller is allowed to see."
    ),
    parameters={
        "type": "object",
        "properties": {
            "patient_id": {
                "type": "string",
                "description": "Patient identifier, e.g. '2013'. Omit to list "
                               "all jobs the caller can see.",
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
                "description": "Maximum number of jobs to return. Defaults to "
                               "10, capped server-side.",
            },
        },
        "required": [],
    },
)


SEARCH_KNOWLEDGE_BASE = ToolSpec(
    name="search_knowledge_base",
    description=(
        "Search BrainSightAI's product documentation, FAQs and neuroimaging "
        "guides. Use this for questions about how VoxelBox works, what a "
        "feature does, acquisition protocols, file formats, or general "
        "neuroimaging concepts. Do not use it to look up the status of a "
        "specific scan; use get_job_status for that."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, phrased as the user's "
                               "underlying question.",
            },
        },
        "required": ["query"],
    },
)


ALL_TOOLS: List[ToolSpec] = [GET_JOB_STATUS, SEARCH_KNOWLEDGE_BASE]

BY_NAME: Dict[str, ToolSpec] = {t.name: t for t in ALL_TOOLS}


def to_openai(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    """OpenAI chat-completions tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def to_anthropic(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    """Anthropic messages tool format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def to_gemini(tools: List[ToolSpec]):
    """
    google-genai Tool format.

    Imported lazily so the module stays importable without the SDK installed.
    """
    from google.genai import types

    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.parameters,
                )
                for t in tools
            ]
        )
    ]
