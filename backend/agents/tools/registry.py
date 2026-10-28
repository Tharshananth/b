"""
Tool Registry for the ReAct Agent.

Centralizes discovery, registration, and dispatch of agent tools.
"""
from typing import Dict, List, Optional
from llm.tools import ToolSpec
from agents.tools.base import BaseAgentTool
from agents.tools.kb_tool import SearchKnowledgeBaseTool
from agents.tools.job_status_tool import GetJobStatusTool
from agents.tools.stage_analyzer_tool import DiagnoseJobStagesTool
from agents.tools.escalation_tool import EscalateToSupportTool


class ToolRegistry:
    """Registry maintaining available agent tools."""

    def __init__(self):
        self._tools: Dict[str, BaseAgentTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register BrainSightAI default tool suite."""
        self.register(GetJobStatusTool())
        self.register(SearchKnowledgeBaseTool())
        self.register(DiagnoseJobStagesTool())
        self.register(EscalateToSupportTool())

    def register(self, tool: BaseAgentTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseAgentTool]:
        """Look up tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseAgentTool]:
        """Return list of all registered tool instances."""
        return list(self._tools.values())

    def get_all_tool_specs(self) -> List[ToolSpec]:
        """Return provider-neutral ToolSpec list for LLM configuration."""
        return [tool.to_tool_spec() for tool in self._tools.values()]


# Default singleton instance
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry singleton."""
    return _registry
