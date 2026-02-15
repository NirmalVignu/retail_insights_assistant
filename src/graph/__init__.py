"""
LangGraph-based Query Processing
State management and workflow orchestration
"""

from .langgraph_agent import LangGraphQueryAgent
from .enhanced_query_resolution import EnhancedQueryResolutionAgent

# Export with both names for compatibility
LangGraphAgent = LangGraphQueryAgent

__all__ = [
    "LangGraphQueryAgent",
    "LangGraphAgent",  # Alias for backward compatibility
    "EnhancedQueryResolutionAgent"
]
