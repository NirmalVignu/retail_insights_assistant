"""
LangGraph-based Multi-Agent System for Complex Query Resolution
Upgraded: Intelligent sub-query decomposition, LLM fallback, and advanced reasoning
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from ..utils.data_processor import get_processor
from ..utils.llm_config import get_llm

# Optional import - requires faiss
try:
    from ..utils.conversation_memory import ConversationMemory
except ImportError:
    ConversationMemory = None

from .enhanced_query_resolution import EnhancedQueryResolutionAgent
from ..utils.response_formatter import ResponseFormatter
from ..utils.prompt_loader import format_prompt, load_prompt
import logging
import json
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangGraphQueryAgent:
    def __init__(self, processor=None, conversation_memory: Optional[ConversationMemory] = None):
        self.processor = processor or get_processor()
        self.memory = conversation_memory
        self.llm = get_llm(temperature=0.1)
        self.enhanced_agent = EnhancedQueryResolutionAgent(self.processor, self.memory)
        self.formatter = ResponseFormatter(self.processor)
        
        # Define tools for intelligent reasoning
        self.tools = self._define_tools()
        # Note: Tools are defined but direct execution is handled in workflow nodes
        self.graph = self._build_graph()
    
    def _define_tools(self) -> List[tool]:
        """Define tools available to the agents for intelligent reasoning"""
        
        @tool
        def query_database(sql_query: str) -> Dict[str, Any]:
            """Execute SQL query on the database"""
            try:
                result = self.processor.query(sql_query)
                return {
                    "success": True,
                    "row_count": len(result) if hasattr(result, '__len__') else 0,
                    "data": result,
                    "error": None
                }
            except Exception as e:
                logger.error(f"Tool query_database failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "data": None
                }
        
        @tool
        def get_table_info() -> Dict[str, Any]:
            """Get information about available tables and columns"""
            try:
                tables = self.processor.list_tables()
                table_info = {}
                for table in tables:
                    columns = self.processor.get_table_columns(table)
                    table_info[table] = columns
                return {
                    "tables": table_info,
                    "table_count": len(tables)
                }
            except Exception as e:
                logger.error(f"Tool get_table_info failed: {e}")
                return {"error": str(e)}
        
        @tool
        def validate_sql(sql_query: str) -> Dict[str, Any]:
            """Validate and optimize SQL query before execution"""
            try:
                # Basic SQL validation checks
                sql_lower = sql_query.lower().strip()
                issues = []
                
                if not sql_lower.startswith("select"):
                    issues.append("Query must start with SELECT")
                
                if "drop" in sql_lower or "delete" in sql_lower or "truncate" in sql_lower:
                    issues.append("Destructive operations not allowed")
                
                if len(issues) > 0:
                    return {"valid": False, "issues": issues}
                
                return {"valid": True, "issues": [], "optimized_query": sql_query}
            except Exception as e:
                return {"valid": False, "error": str(e)}
        
        return [query_database, get_table_info, validate_sql]

    def _build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("analyze_query", self._node_analyze_query)
        workflow.add_node("decompose_query", self._node_decompose_query)
        workflow.add_node("extract_data", self._node_extract_data)
        workflow.add_node("llm_analysis", self._node_llm_analysis)
        workflow.add_node("validate_results", self._node_validate_results)
        workflow.add_node("refine_query", self._node_refine_query)
        workflow.add_node("format_response", self._node_format_response)
        # Edges: strictly sequential, only one path at a time
        workflow.add_edge("analyze_query", "decompose_query")
        workflow.add_edge("decompose_query", "extract_data")
        workflow.add_edge("extract_data", "validate_results")
        # Validation: decide next step
        def validation_router(state):
            if state.get("needs_refine", False) and state.get("refinement_count", 0) < 2:
                return "refine_query"
            elif state.get("llm_fallback"):
                return "llm_analysis"
            else:
                return "format_response"
        workflow.add_conditional_edges(
            "validate_results",
            validation_router,
            {"refine_query": "refine_query", "llm_analysis": "llm_analysis", "format_response": "format_response"}
        )
        workflow.add_edge("refine_query", "extract_data")
        workflow.add_edge("llm_analysis", "format_response")
        workflow.add_edge("format_response", END)
        workflow.set_entry_point("analyze_query")
        return workflow.compile()

    def _ensure_flat_state(self, state: Dict[str, Any], node_name: str) -> Dict[str, Any]:
        if not isinstance(state, dict):
            logger.error(f"{node_name}: State is not a dict! Actual type: {type(state)}")
            raise ValueError(f"{node_name}: State must be a dict, got {type(state)}")
        if any(isinstance(v, dict) and k == "root" for k, v in state.items()):
            logger.error(f"{node_name}: State contains a nested dict at root key!")
            raise ValueError(f"{node_name}: State contains a nested dict at root key!")
        logger.debug(f"{node_name}: State type OK. Keys: {list(state.keys())}")
        return state

    def _node_analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_query = state.get("user_query", "")
        logger.info(f"LangGraph: Analyzing query: {user_query}")
        
        try:
            # Use enhanced agent for query resolution
            analysis = self.enhanced_agent.resolve_query(user_query)
            state["query_analysis"] = analysis
            state["conversation_context"] = self.memory.messages if self.memory else []
            state["analyzed"] = True
            logger.info(f"Query analyzed successfully: {analysis.get('parsed_intent')}")
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["error"] = str(e)
            state["analyzed"] = False
            # Provide fallback analysis
            state["query_analysis"] = {
                "query_type": "analytical",
                "parsed_intent": user_query,
                "confidence_score": 0.3,
                "error": str(e)
            }
        
        return self._ensure_flat_state(state, "analyze_query")

    def _node_decompose_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        analysis = state.get("query_analysis", {})
        query_type = analysis.get("query_type", "analytical")
        logger.info(f"LangGraph: Decomposing query type: {query_type}")
        if query_type in ["comparison", "timeseries", "custom"]:
            try:
                decomposition_prompt = load_prompt("query_decomposition_prompt")
                messages = [
                    SystemMessage(content=decomposition_prompt),
                    HumanMessage(content=f"Decompose this complex query into simple sub-queries:\n\n{analysis.get('parsed_intent')}\n\nReturn ONLY a JSON array of strings.")
                ]
                response = self.llm.invoke(messages)
                sub_queries = json.loads(response.content)
                state["sub_queries"] = sub_queries
                state["decomposed"] = True
            except Exception as e:
                logger.warning(f"Decomposition failed: {e}")
                state["sub_queries"] = []
                state["decomposed"] = False
        else:
            state["sub_queries"] = []
            state["decomposed"] = False
        return self._ensure_flat_state(state, "decompose_query")

    def _node_extract_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        analysis = state.get("query_analysis", {})
        sub_queries = state.get("sub_queries", [])
        logger.info(f"LangGraph: Extracting data for: {analysis.get('parsed_intent')}")
        
        try:
            if sub_queries:
                logger.info(f"Executing {len(sub_queries)} sub-queries...")
                all_results = []
                for idx, sub_q in enumerate(sub_queries):
                    try:
                        sub_analysis = self.enhanced_agent.resolve_query(sub_q)
                        sql = self._build_sql(sub_analysis)
                        logger.info(f"Sub-query {idx+1} SQL: {sql}")
                        result = self.processor.query(sql)
                        all_results.append({
                            "data": result, 
                            "sql": sql, 
                            "row_count": len(result) if hasattr(result, '__len__') else 0, 
                            "sub_query": sub_q,
                            "success": True
                        })
                    except Exception as sub_e:
                        logger.error(f"Sub-query {idx+1} failed: {sub_e}")
                        all_results.append({
                            "error": str(sub_e),
                            "sub_query": sub_q,
                            "success": False
                        })
                
                # Combine results intelligently
                combined = self._combine_results(all_results)
                state["extracted_data"] = {
                    "sub_results": all_results, 
                    "combined": combined,
                    "row_count": combined.get("row_count", 0)
                }
            else:
                sql = self._build_sql(analysis)
                logger.info(f"Executing SQL: {sql}")
                result = self.processor.query(sql)
                state["extracted_data"] = {
                    "data": result, 
                    "row_count": len(result) if hasattr(result, '__len__') else 0, 
                    "sql": sql,
                    "success": True
                }
            
            logger.info(f"Data extraction completed: {state['extracted_data'].get('row_count', 0)} rows")
            
        except Exception as e:
            logger.error(f"LangGraph: Data extraction failed: {e}")
            state["extracted_data"] = {"error": str(e), "success": False}
        
        return self._ensure_flat_state(state, "extract_data")

    def _node_llm_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_query = state.get("user_query", "")
        extracted = state.get("extracted_data", {})
        analysis = state.get("query_analysis", {})
        logger.info("LangGraph: Extracting insights from data...")
        try:
            # Prepare data summary for LLM
            data_df = extracted.get("data")
            if data_df is not None and isinstance(data_df, pd.DataFrame):
                # For better LLM understanding, show full data more clearly
                if len(data_df) <= 50:
                    # Small dataset - show everything
                    data_display = data_df.to_string(index=False)
                    data_size_info = f"Complete dataset ({len(data_df)} rows):"
                else:
                    # Large dataset - show top 30 rows
                    data_display = data_df.head(30).to_string(index=False)
                    data_size_info = f"Top 30 of {len(data_df)} total rows:"
                
                # Add descriptive stats only for large datasets
                stats_info = ""
                if len(data_df) > 50:
                    numeric_cols = data_df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        stats_info = f"\n\nNumeric Column Statistics:\n{data_df[numeric_cols].describe().to_string()}"
            else:
                data_display = str(extracted.get("sub_results", []))
                data_size_info = "Query Results:"
                stats_info = ""
            
            llm_analysis_template = load_prompt("llm_analysis_prompt")
            insight_prompt = llm_analysis_template.format(
                user_query=user_query,
                data_size_info=data_size_info,
                data_display=data_display,
                stats_info=stats_info,
                aggregations=analysis.get('aggregations', []),
                groupby=analysis.get('groupby', []),
                orderby=analysis.get('orderby', {})
            )
            
            messages = [
                SystemMessage(content=insight_prompt)
            ]
            response = self.llm.invoke(messages)
            state["llm_analysis"] = response.content
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            state["llm_analysis"] = f"Analysis: {extracted.get('data', 'No data available')}"
        return self._ensure_flat_state(state, "llm_analysis")

    def _node_validate_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        data = state.get("extracted_data", {})
        logger.info("LangGraph: Validating results...")
        if data.get("error"):
            state["needs_refine"] = True
            state["validation_message"] = data["error"]
            state["llm_fallback"] = True
        elif data.get("row_count", 0) == 0:
            state["needs_refine"] = True
            state["validation_message"] = "No data found."
            state["llm_fallback"] = True
        else:
            state["needs_refine"] = False
            state["validation_message"] = f"Valid: {data.get('row_count')} rows."
            # Always analyze successful queries with LLM for insights
            state["llm_fallback"] = True
        return self._ensure_flat_state(state, "validate_results")

    def _node_refine_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_query = state.get("user_query", "")
        prev_message = state.get("validation_message", "")
        logger.info(f"LangGraph: Refining query due to: {prev_message}")
        refined = self.enhanced_agent.resolve_query(f"{user_query}. Previous error: {prev_message}")
        state["query_analysis"] = refined
        state["refinement_count"] = state.get("refinement_count", 0) + 1
        return self._ensure_flat_state(state, "refine_query")

    def _node_format_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        data = state.get("extracted_data", {})
        analysis = state.get("query_analysis", {})
        llm_analysis = state.get("llm_analysis", "")
        logger.info("LangGraph: Formatting response...")
        
        # Use LLM insights as primary summary when available
        if llm_analysis and llm_analysis.strip():
            summary = llm_analysis
        else:
            summary = state.get("validation_message", "No analysis available")
        
        formatted = self.formatter.format_response(
            summary=summary,
            data=data.get("data"),
            query_result=analysis,
            confidence=analysis.get("confidence_score", 0.5)
        )
        state["formatted_response"] = formatted
        return self._ensure_flat_state(state, "format_response")

    def _should_decompose(self, state: Dict[str, Any]) -> str:
        if state.get("decomposed"):
            return "decompose"
        return "simple"

    def _should_llm_fallback(self, state: Dict[str, Any]) -> str:
        if state.get("llm_fallback"):
            return "llm"
        return "ok"

    def _should_refine(self, state: Dict[str, Any]) -> str:
        if state.get("needs_refine", False) and state.get("refinement_count", 0) < 2:
            return "refine"
        return "format"

    def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Intelligently combine results from multiple sub-queries"""
        combined_data = []
        total_rows = 0
        successful_queries = 0
        errors = []
        
        for result in results:
            if result.get("success"):
                data = result.get("data", [])
                if hasattr(data, '__iter__') and not isinstance(data, str):
                    if isinstance(data, pd.DataFrame):
                        combined_data.append(data)
                        total_rows += len(data)
                    else:
                        combined_data.extend(data)
                        total_rows += len(data)
                successful_queries += 1
            else:
                errors.append(result.get("error", "Unknown error"))
        
        # Merge DataFrames if possible
        if combined_data and all(isinstance(d, pd.DataFrame) for d in combined_data):
            try:
                merged_df = pd.concat(combined_data, ignore_index=True)
                return {
                    "data": merged_df,
                    "row_count": len(merged_df),
                    "successful_queries": successful_queries,
                    "errors": errors
                }
            except Exception as e:
                logger.warning(f"Failed to merge DataFrames: {e}")
        
        return {
            "data": combined_data,
            "row_count": total_rows,
            "successful_queries": successful_queries,
            "errors": errors
        }
    
    def _build_sql(self, analysis: Dict[str, Any]) -> str:
        """Build SQL from query specification with intelligent enhancements"""
        table = analysis.get("primary_table", "")
        entities = analysis.get("entities", [])
        groupby = analysis.get("groupby", [])
        filters = analysis.get("filters", {})
        aggregations = analysis.get("aggregations", [])
        orderby = analysis.get("orderby", {})
        
        select_clause = "*"
        groupby_clause = ""
        orderby_clause = ""
        
        # Detect if groupby is a date column for monthly/time-based trend
        date_groupby = None
        for col in groupby:
            if col.lower() in ["date", "order date", "order_date", "transaction_date"]:
                date_groupby = col
                break
        
        if date_groupby:
            # Use strftime with explicit CAST for date columns
            month_expr = f"strftime('%Y-%m', CAST({date_groupby} AS DATE)) as month"
            
            # Build select with aggregations if specified
            select_parts = [month_expr]
            for entity in entities:
                if entity != date_groupby:
                    if aggregations:
                        # Apply aggregations to numeric columns
                        for agg in aggregations:
                            select_parts.append(f"{agg.upper()}({entity}) as {agg}_{entity}")
                    else:
                        select_parts.append(entity)
            
            select_clause = ", ".join(select_parts)
            groupby_clause = f" GROUP BY month"
            orderby_clause = " ORDER BY month ASC"
        else:
            # Standard SQL building
            if entities:
                select_parts = []
                for entity in entities:
                    if aggregations:
                        for agg in aggregations:
                            select_parts.append(f"{agg.upper()}({entity}) as {agg}_{entity}")
                    else:
                        select_parts.append(entity)
                select_clause = ", ".join(select_parts) if select_parts else ", ".join(entities)
            else:
                select_clause = "*"
            
            if groupby:
                groupby_clause = " GROUP BY " + ", ".join(groupby)
            
            if orderby:
                order_parts = [f"{col} {direction}" for col, direction in orderby.items()]
                orderby_clause = " ORDER BY " + ", ".join(order_parts)
        
        # Build final SQL
        sql = f"SELECT {select_clause} FROM {table}"
        
        if filters:
            where_conditions = [f"{k} = '{v}'" for k, v in filters.items()]
            sql += " WHERE " + " AND ".join(where_conditions)
        
        sql += groupby_clause
        sql += orderby_clause
        
        limit = analysis.get("limit", 100)
        sql += f" LIMIT {limit}"
        
        logger.info(f"Built SQL: {sql}")
        return sql

    def process_query(self, user_query: str) -> Dict[str, Any]:
        initial_state = {"user_query": user_query, "refinement_count": 0}
        final_state = self.graph.invoke(initial_state)
        return {
            "user_query": user_query,
            "query_resolution": final_state.get("query_analysis"),
            "extracted_data": final_state.get("extracted_data"),
            "formatted_response": final_state.get("formatted_response"),
            "confidence_score": final_state.get("query_analysis", {}).get("confidence_score", 0.5),
            "success": not final_state.get("needs_refine", False)
        }
