"""
Enhanced Query Resolution Agent with Conversation Context
Supports complex queries, context awareness, and confidence scoring
Includes intelligent column mapping and fuzzy matching
"""

import json
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from ..utils.data_processor import get_processor
from ..utils.llm_config import get_llm

# Optional import - requires faiss
try:
    from ..utils.conversation_memory import ConversationMemory
except ImportError:
    ConversationMemory = None

from ..utils.prompt_loader import format_prompt
import logging
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedQueryResolutionAgent:
    """
    Advanced query resolution with conversation context and confidence scoring
    Includes intelligent fuzzy column matching
    """
    
    def __init__(self, processor=None, conversation_memory: Optional[ConversationMemory] = None):
        self.llm = get_llm(temperature=0.1)
        self.processor = processor or get_processor()
        self.memory = conversation_memory
        self.column_cache = {}  # Cache for table schemas
    
    def fuzzy_match_column(self, user_term: str, available_columns: List[str], threshold: float = 0.6) -> Optional[str]:
        """
        Intelligently match user's column reference to actual column names using fuzzy matching
        
        Args:
            user_term: What the user mentioned (e.g., "sales", "total sales", "revenue")
            available_columns: List of actual column names
            threshold: Similarity threshold (0.6 = 60%)
        
        Returns:
            Best matching column name or None
        """
        if not available_columns or not user_term:
            return None
        
        user_term_lower = user_term.lower().strip()
        best_match = None
        best_ratio = 0
        
        # Exact match (case-insensitive)
        for col in available_columns:
            if col.lower() == user_term_lower:
                return col
        
        # Partial match - check if column name contains user term or vice versa
        for col in available_columns:
            col_lower = col.lower()
            if user_term_lower in col_lower or col_lower in user_term_lower:
                ratio = SequenceMatcher(None, user_term_lower, col_lower).ratio()
                if ratio > best_ratio and ratio >= threshold:
                    best_match = col
                    best_ratio = ratio
        
        # If no partial match, use fuzzy sequence matching
        if not best_match:
            for col in available_columns:
                ratio = SequenceMatcher(None, user_term_lower, col.lower()).ratio()
                if ratio > best_ratio and ratio >= threshold:
                    best_match = col
                    best_ratio = ratio
        
        logger.info(f"Fuzzy match: '{user_term}' -> '{best_match}' (ratio: {best_ratio:.2f})")
        return best_match
    
    def get_table_schema(self, table_name: str) -> Dict[str, List[str]]:
        """
        Get schema information for a table
        Returns: {"columns": ["col1", "col2", ...], "types": {...}}
        """
        if table_name in self.column_cache:
            return self.column_cache[table_name]
        
        try:
            # Get actual columns from the table
            columns = self.processor.get_table_columns(table_name)
            schema = {
                "columns": columns if columns else [],
                "count": len(columns) if columns else 0
            }
            self.column_cache[table_name] = schema
            return schema
        except Exception as e:
            logger.error(f"Failed to get schema for {table_name}: {e}")
            return {"columns": [], "count": 0}
    
    def build_schema_info(self) -> str:
        """Build detailed schema information for all tables for the LLM"""
        tables = self.processor.list_tables()
        schema_info = "Available Tables and Columns:\n"
        
        for table in tables:
            schema = self.get_table_schema(table)
            columns = schema.get("columns", [])
            schema_info += f"\n{table}:\n"
            schema_info += f"  Columns: {', '.join(columns)}\n"
        
        return schema_info
    
    def resolve_query(self, user_query: str, rag_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve query with conversation context and confidence scoring
        Uses intelligent fuzzy column matching
        
        Returns:
            {
                "query_type": "summary|analytical|comparison|timeseries",
                "primary_table": "table_name",
                "entities": ["column1", "column2"],
                "filters": {"column": "value"},
                "aggregations": ["sum", "count", "avg"],
                "groupby": ["column1"],
                "orderby": {"column": "DESC"},
                "limit": 10,
                "parsed_intent": "human readable intent",
                "confidence_score": 0.95,
                "requires_context": bool,
                "suggested_visualizations": ["bar_chart", "line_chart"]
            }
        """
        
        try:
            tables = self.processor.list_tables()
            tables_info = "\n".join(tables) if tables else "No tables loaded"
            
            # Build detailed schema information
            schema_info = self.build_schema_info()
            
            # Build context-aware prompt
            context_part = ""
            if rag_context:
                context_part = f"\n\nPrevious conversation context:\n{rag_context}"
            
            # Load prompt from file
            system_prompt = format_prompt(
                "query_resolution_prompt",
                schema_info=schema_info,
                context_part=context_part
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User Query: {user_query}")
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse response
            try:
                # Extract JSON from response
                response_text = response.content.strip()
                
                # Handle markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                
                result = json.loads(response_text.strip())
                
                # Ensure all required fields
                result.setdefault("confidence_score", 0.7)
                result.setdefault("requires_context", False)
                result.setdefault("suggested_visualizations", [])
                result.setdefault("limit", 100)
                
                # Post-process: Apply fuzzy matching to resolve columns to actual names
                result = self._resolve_columns_to_actuals(result)
                
                logger.info(f"Query resolved with confidence: {result['confidence_score']}")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response: {e}")
                # Return default structured response
                return {
                    "query_type": "analytical",
                    "primary_table": tables[0] if tables else "unknown",
                    "entities": [],
                    "filters": {},
                    "aggregations": [],
                    "parsed_intent": user_query,
                    "confidence_score": 0.3,
                    "requires_context": False,
                    "suggested_visualizations": []
                }
        
        except Exception as e:
            logger.error(f"Query resolution failed: {e}")
            return {
                "error": str(e),
                "confidence_score": 0.0,
                "parsed_intent": user_query
            }
    
    def _resolve_columns_to_actuals(self, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process query result to resolve column references to actual column names
        using fuzzy matching
        """
        try:
            primary_table = query_result.get("primary_table")
            if not primary_table:
                return query_result
            
            schema = self.get_table_schema(primary_table)
            actual_columns = schema.get("columns", [])
            
            if not actual_columns:
                return query_result
            
            # Resolve entities (columns to select)
            if "entities" in query_result and query_result["entities"]:
                resolved_entities = []
                for entity in query_result["entities"]:
                    matched = self.fuzzy_match_column(entity, actual_columns, threshold=0.5)
                    if matched:
                        resolved_entities.append(matched)
                        logger.info(f"Resolved entity '{entity}' -> '{matched}'")
                    else:
                        # If no match found, keep original
                        if entity not in resolved_entities:
                            resolved_entities.append(entity)
                query_result["entities"] = resolved_entities
            
            # Resolve groupby columns
            if "groupby" in query_result and query_result["groupby"]:
                resolved_groupby = []
                for col in query_result["groupby"]:
                    matched = self.fuzzy_match_column(col, actual_columns, threshold=0.5)
                    if matched:
                        resolved_groupby.append(matched)
                        logger.info(f"Resolved groupby '{col}' -> '{matched}'")
                    else:
                        if col not in resolved_groupby:
                            resolved_groupby.append(col)
                query_result["groupby"] = resolved_groupby
            
            # Resolve orderby columns
            if "orderby" in query_result and query_result["orderby"]:
                resolved_orderby = {}
                for col, direction in query_result["orderby"].items():
                    matched = self.fuzzy_match_column(col, actual_columns, threshold=0.5)
                    if matched:
                        resolved_orderby[matched] = direction
                        logger.info(f"Resolved orderby '{col}' -> '{matched}'")
                    else:
                        resolved_orderby[col] = direction
                query_result["orderby"] = resolved_orderby
            
            # Resolve filter columns
            if "filters" in query_result and query_result["filters"]:
                resolved_filters = {}
                for col, value in query_result["filters"].items():
                    matched = self.fuzzy_match_column(col, actual_columns, threshold=0.5)
                    if matched:
                        resolved_filters[matched] = value
                        logger.info(f"Resolved filter '{col}' -> '{matched}'")
                    else:
                        resolved_filters[col] = value
                query_result["filters"] = resolved_filters
            
            return query_result
        
        except Exception as e:
            logger.error(f"Error resolving columns to actuals: {e}")
            return query_result
    
    def evaluate_confidence(self, query_result: Dict[str, Any]) -> float:
        """
        Evaluate confidence in query interpretation
        Factors: confidence_score, query complexity, data availability
        
        Returns:
            Confidence score 0-1
        """
        base_score = query_result.get("confidence_score", 0.5)
        
        # Penalize if complex fields are missing
        missing_factors = 0
        if not query_result.get("aggregations"):
            missing_factors += 0.1
        if not query_result.get("filters"):
            missing_factors += 0.05
        
        final_score = max(0, min(1, base_score - missing_factors))
        return final_score
    
    def suggest_followup_queries(self, query_result: Dict[str, Any]) -> List[str]:
        """
        Suggest relevant follow-up questions based on current query
        
        Args:
            query_result: Parsed query result
            
        Returns:
            List of suggested follow-up questions
        """
        suggestions = []
        query_type = query_result.get("query_type", "")
        
        if query_type == "summary":
            suggestions = [
                "Can you break down by region?",
                "What's the trend over time?",
                "Compare with previous period?"
            ]
        elif query_type == "analytical":
            suggestions = [
                "Which segment performed best?",
                "What's the growth rate?",
                "Show comparison with competitors?"
            ]
        elif query_type == "comparison":
            suggestions = [
                "What are the key drivers?",
                "Which variables had most impact?",
                "Forecast for next period?"
            ]
        
        return suggestions


if __name__ == "__main__":
    # Example usage
    agent = EnhancedQueryResolutionAgent()
    
    # Test queries
    queries = [
        "What were total sales in Q4?",
        "Show me YoY growth for North region in Q3 vs Q4",
        "Which products had declining trends?"
    ]
    
    for q in queries:
        result = agent.resolve_query(q)
        confidence = agent.evaluate_confidence(result)
        print(f"Query: {q}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Type: {result.get('query_type')}")
        print(f"Suggestions: {agent.suggest_followup_queries(result)}\n")
