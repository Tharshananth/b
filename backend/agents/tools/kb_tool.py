"""
Knowledge Base Search Tool for ReAct Agent.

Searches VoxelBox platform docs, neuroimaging FAQs, and protocols via ChromaDB RAG.
"""
from typing import Any, Dict, List, Optional, Tuple
from agents.tools.base import BaseAgentTool
from services.job_status import TenantScope


class SearchKnowledgeBaseTool(BaseAgentTool):
    """Tool for searching BrainSightAI's documentation and knowledge base."""

    name = "search_knowledge_base"
    description = (
        "Search BrainSightAI's product documentation, FAQs, acquisition protocols, "
        "and neuroimaging guides. Use this for questions about how VoxelBox works, "
        "what a feature does, MRI acquisition parameters, file formats, or general "
        "neuroimaging concepts. Do not use it to look up specific scan jobs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, phrased as the user's underlying technical question.",
            },
        },
        "required": ["query"],
    }

    def _run(
        self,
        args: Dict[str, Any],
        scope: Optional[TenantScope],
        context: Dict[str, Any],
    ) -> Tuple[Any, str, List[str], List[Dict[str, str]], Any]:
        query = (args.get("query") or "").strip()
        if not query:
            return (
                {"error": "Empty search query provided."},
                "empty query",
                [],
                [],
                None,
            )

        from vector_db.retriever import get_retriever
        retriever = get_retriever()
        ctx = retriever.retrieve_context(query)
        sources = ctx.get("sources", []) or []
        summary = f"{len(sources)} source(s) found"

        return ctx.get("context", ""), summary, [], sources, ctx
