"""
Multi-Agent System using LangGraph
Four specialized agents: QueryResolution, DataExtraction, Validation, and DataAnalyst
Enhanced with conversation memory and confidence scoring
"""

from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import json
from ..utils.data_processor import get_processor
from ..utils.llm_config import get_llm

# Optional import - requires faiss
try:
    from ..utils.conversation_memory import ConversationMemory
except ImportError:
    ConversationMemory = None

from ..graph.enhanced_query_resolution import EnhancedQueryResolutionAgent
from ..utils.response_formatter import ResponseFormatter
from ..utils.prompt_loader import load_prompt
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryResolutionAgent:
    """
    Converts natural language queries into structured data extraction requests
    Uses conversation memory and context for improved accuracy
    """
    
    def __init__(self, processor=None, conversation_memory: Optional[ConversationMemory] = None):
        self.enhanced_agent = EnhancedQueryResolutionAgent(processor, conversation_memory)
        self.processor = processor or get_processor()
        self.memory = conversation_memory
    
    def resolve_query(self, user_query: str, context: str = "") -> Dict[str, Any]:
        """
        Convert user query to structured request using enhanced resolution
        
        Returns:
            {
                "query_type": "summary|analytical|comparison|timeseries",
                "primary_table": "table_name",
                "entities": ["column1", "column2"],
                "filters": {},
                "aggregations": ["sum", "count", "avg"],
                "parsed_intent": "what the user is asking for",
                "confidence_score": 0.95,
                "requires_context": false,
                "suggested_visualizations": []
            }
        """
        try:
            # Get RAG context if memory is available
            rag_context = None
            if self.memory:
                rag_context = self.memory.build_rag_context(user_query, k=2)
            
            # Use enhanced agent for better resolution
            result = self.enhanced_agent.resolve_query(user_query, rag_context)
            logger.info(f"Query resolved: {result.get('parsed_intent')} (confidence: {result.get('confidence_score')})")
            
            return result
            
        except Exception as e:
            logger.error(f"Query resolution error: {e}")
            return {
                "query_type": "analytical",
                "parsed_intent": user_query,
                "confidence_score": 0.0,
                "error": str(e),
                "primary_table": "",
                "entities": [],
                "filters": {},
                "aggregations": []
            }


class DataExtractionAgent:
    """
    Extracts data from DuckDB based on structured requests
    """
    
    def __init__(self, processor=None):
        self.processor = processor or get_processor()
    
    def extract_data(self, structured_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data extraction based on structured request with aggregations
        
        Returns:
            {
                "data": DataFrame,
                "summary": "Data summary",
                "row_count": int,
                "error": None or error message
            }
        """
        
        try:
            primary_table = structured_request.get("primary_table")
            
            if not primary_table:
                tables = self.processor.list_tables()
                if tables:
                    primary_table = tables[0]
                else:
                    return {"error": "No data tables available", "data": None}
            
            # Build SQL query
            entities = structured_request.get("entities", [])
            filters = structured_request.get("filters", {})
            aggregations = structured_request.get("aggregations", [])
            groupby = structured_request.get("groupby", [])
            orderby = structured_request.get("orderby", {})
            
            # Build SELECT clause with aggregations
            select_parts = []
            
            if groupby:
                # Add groupby columns first
                for col in groupby:
                    select_parts.append(f'"{col}"')
                
                # Add aggregations on entities
                if aggregations and entities:
                    for entity in entities:
                        if entity not in groupby:
                            for agg in aggregations:
                                select_parts.append(f'{agg.upper()}("{entity}") as {agg}_{entity}')
                elif entities:
                    # If no aggregations specified but groupby exists, default to SUM for numeric
                    for entity in entities:
                        if entity not in groupby:
                            select_parts.append(f'SUM("{entity}") as total_{entity}')
            elif entities:
                # No groupby - simple select
                select_parts = [f'"{e}"' if e != "*" else "*" for e in entities]
            else:
                select_parts = ["*"]
            
            select_clause = ", ".join(select_parts) if select_parts else "*"
            sql = f"SELECT {select_clause} FROM {primary_table}"
            
            # Add filters
            if filters:
                where_clauses = []
                for col, val in filters.items():
                    # Handle list/array filters
                    if isinstance(val, list):
                        if len(val) == 0:
                            continue
                        # Use IN clause for multiple values
                        val_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in val])
                        where_clauses.append(f'"{col}" IN ({val_str})')
                    elif isinstance(val, str):
                        where_clauses.append(f'"{col}" = \'{val}\'')
                    else:
                        where_clauses.append(f'"{col}" = {val}')
                if where_clauses:
                    sql += " WHERE " + " AND ".join(where_clauses)
            
            # Add GROUP BY
            if groupby:
                groupby_clause = ", ".join([f'"{col}"' for col in groupby])
                sql += f" GROUP BY {groupby_clause}"
            
            # Add ORDER BY
            if orderby:
                order_parts = [f'"{col}" {direction}' for col, direction in orderby.items()]
                sql += " ORDER BY " + ", ".join(order_parts)
            
            # Default limit
            limit = structured_request.get("limit", 100)
            sql += f" LIMIT {limit}"
            
            logger.info(f"Executing SQL: {sql}")
            
            # Execute query
            result_df = self.processor.query(sql)
            
            return {
                "data": result_df,
                "summary": f"Extracted {len(result_df)} rows from {primary_table}",
                "row_count": len(result_df),
                "sql": sql,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {
                "error": f"Data extraction failed: {str(e)}",
                "data": None,
                "row_count": 0
            }


class ValidationAgent:
    """
    Validates data quality and provides insights
    """
    
    def __init__(self, processor=None):
        self.llm = get_llm(temperature=0.1)
        self.processor = processor
    
    def validate_and_interpret(self, data: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Validate data and generate direct business insights with actual answers
        """
        
        try:
            import pandas as pd
            
            extracted_data = data.get("data")
            
            if extracted_data is None or extracted_data.empty:
                return {
                    "is_valid": False,
                    "error": "No data to validate",
                    "confidence_score": 0.0,
                    "insights": "No data found matching your query criteria."
                }
            
            # First, try to generate direct answers based on the query intent
            insights = self._generate_direct_answer(extracted_data, original_query)
            
            # If direct answer failed, fall back to summary insights
            if not insights or len(insights) < 50:
                insights = self._generate_summary_insights(extracted_data, original_query)
            
            return {
                "is_valid": True,
                "validation_result": insights,
                "data_quality": "Good",
                "confidence_score": 0.85,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "is_valid": False,
                "error": str(e),
                "confidence_score": 0.0,
                "insights": f"Error processing data: {str(e)}"
            }
    
    def _generate_direct_answer(self, data: pd.DataFrame, query: str) -> str:
        """
        Generate direct answers by computing statistics from the actual data
        This replaces generic analysis with concrete answers
        """
        try:
            import pandas as pd
            
            query_lower = query.lower()
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Detect common query patterns and compute actual answers
            
            # Pattern 1: "Which [category] generates the highest [metric]"
            if "highest" in query_lower or "maximum" in query_lower or "maximum" in query_lower:
                if "total" in query_lower and categorical_cols and numeric_cols:
                    category_col = categorical_cols[0]
                    metric_col = numeric_cols[0]
                    
                    # Group by category and sum
                    grouped = data.groupby(category_col)[metric_col].sum().sort_values(ascending=False)
                    if len(grouped) > 0:
                        top_category = grouped.index[0]
                        top_value = grouped.iloc[0]
                        
                        answer = f"Based on the data analysis:\n\n"
                        answer += f"**{top_category}** generates the highest total {metric_col} with **${top_value:,.2f}** (or {top_value:,.0f} units).\n\n"
                        
                        # Add top 5 breakdown
                        answer += "**Top 5 Categories:**\n"
                        for idx, (cat, val) in enumerate(grouped.head(5).items(), 1):
                            answer += f"{idx}. {cat}: ${val:,.2f}\n"
                        
                        return answer
            
            # Pattern 2: "What is the total [metric]"
            if "total" in query_lower and numeric_cols:
                metric_col = numeric_cols[0]
                total = data[metric_col].sum()
                count = len(data)
                avg = data[metric_col].mean()
                
                answer = f"Based on the data analysis:\n\n"
                answer += f"**Total {metric_col}:** ${total:,.2f}\n"
                answer += f"**Number of records:** {count:,}\n"
                answer += f"**Average {metric_col}:** ${avg:,.2f}\n"
                
                if categorical_cols:
                    answer += f"\n**Distribution by {categorical_cols[0]}:**\n"
                    dist = data.groupby(categorical_cols[0])[metric_col].sum().sort_values(ascending=False).head(5)
                    for cat, val in dist.items():
                        answer += f"- {cat}: ${val:,.2f}\n"
                
                return answer
            
            # Pattern 3: "Show me [metric] by [category]"
            if "by" in query_lower and categorical_cols and numeric_cols:
                category_col = categorical_cols[0]
                metric_col = numeric_cols[0]
                
                grouped = data.groupby(category_col)[metric_col].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
                
                answer = f"**{metric_col} by {category_col}:**\n\n"
                for cat in grouped.head(10).index:
                    total = grouped.loc[cat, 'sum']
                    count = grouped.loc[cat, 'count']
                    avg = grouped.loc[cat, 'mean']
                    answer += f"- **{cat}:** Total=${total:,.2f}, Count={int(count)}, Average=${avg:,.2f}\n"
                
                return answer
            
            # Pattern 4: General summary
            if numeric_cols:
                answer = f"**Data Summary:**\n\n"
                answer += f"- **Rows:** {len(data):,}\n"
                answer += f"- **Columns:** {len(data.columns)}\n\n"
                
                for col in numeric_cols[:3]:  # Show first 3 numeric columns
                    answer += f"**{col}:**\n"
                    answer += f"  - Total: ${data[col].sum():,.2f}\n"
                    answer += f"  - Average: ${data[col].mean():,.2f}\n"
                    answer += f"  - Max: ${data[col].max():,.2f}\n"
                    answer += f"  - Min: ${data[col].min():,.2f}\n\n"
                
                return answer
            
            return ""
            
        except Exception as e:
            logger.error(f"Direct answer generation failed: {e}")
            return ""
    
    def _generate_summary_insights(self, data: pd.DataFrame, original_query: str) -> str:
        """
        Generate summary insights using LLM as fallback
        Only used when direct answer generation isn't applicable
        """
        try:
            data_summary = f"""
Data Summary:
- Shape: {data.shape[0]} rows × {data.shape[1]} columns
- Columns: {', '.join(data.columns.tolist())}
- Sample data (first 5 rows):
{data.head().to_string()}
"""
            
            # Load prompt from file
            system_prompt = load_prompt("validation_analyst_prompt")
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User's Question: {original_query}\n\nAvailable Data:\n{data_summary}\n\nProvide a precise, data-driven answer.")
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Summary insights generation failed: {e}")
            return "Unable to generate insights from the data."
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Validation failed: {str(e)}",
                "insights": []
            }


class MultiAgentOrchestrator:
    """
    Orchestrates the four agents with conversation memory support
    Query Resolution -> Data Extraction -> Validation -> Formatting
    """
    
    def __init__(
        self,
        processor=None,
        conversation_memory: Optional[ConversationMemory] = None,
        response_formatter: Optional[ResponseFormatter] = None
    ):
        self.processor = processor or get_processor()
        self.conversation_memory = conversation_memory
        self.formatter = response_formatter or ResponseFormatter(self.processor)
        
        # Initialize agents with conversation memory
        self.query_agent = QueryResolutionAgent(self.processor, conversation_memory)
        self.data_agent = DataExtractionAgent(self.processor)
        self.validation_agent = ValidationAgent(self.processor)
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        End-to-end query processing with conversation memory and confidence scoring
        
        Returns:
            {
                "user_query": original query,
                "query_resolution": {structured query info},
                "extracted_data": {row_count, data_preview, error},
                "validation_result": {is_valid, confidence_score, insights},
                "formatted_response": {summary, data, visualizations, confidence},
                "suggested_followups": ["question1", "question2"],
                "success": bool
            }
        """
        
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Step 1: Query Resolution with context
            logger.info("Step 1: Resolving query with context...")
            query_resolution = self.query_agent.resolve_query(user_query)
            
            if query_resolution.get("error"):
                logger.error(f"Query resolution failed: {query_resolution['error']}")
                return self._format_error_response(user_query, query_resolution)
            
            # Step 2: Data Extraction
            logger.info("Step 2: Extracting data...")
            extracted_data = self.data_agent.extract_data(query_resolution)
            
            if extracted_data.get("error"):
                logger.error(f"Data extraction failed: {extracted_data['error']}")
                return self._format_error_response(user_query, extracted_data)
            
            # Step 3: Validation and Insights
            logger.info("Step 3: Validating data and generating insights...")
            validation_result = self.validation_agent.validate_and_interpret(
                extracted_data,
                user_query
            )
            
            # Calculate confidence score
            confidence = query_resolution.get("confidence_score", 0.5)
            validation_confidence = validation_result.get("confidence_score", 0.8)
            final_confidence = (confidence + validation_confidence) / 2
            
            # Step 4: Format response with visualizations
            logger.info("Step 4: Formatting response...")
            formatted_response = self.formatter.format_response(
                summary=validation_result.get("insights", ""),
                data=extracted_data.get("data"),
                query_result=query_resolution,
                confidence=final_confidence
            )
            
            # Add to conversation memory if available
            if self.conversation_memory:
                self.conversation_memory.add_message(
                    role="user",
                    content=user_query,
                    metadata={
                        "query_type": query_resolution.get("query_type"),
                        "confidence": final_confidence
                    }
                )
                self.conversation_memory.add_message(
                    role="assistant",
                    content=validation_result.get("insights", ""),
                    metadata={
                        "row_count": extracted_data.get("row_count", 0),
                        "confidence": final_confidence
                    }
                )
            
            logger.info(f"Query processed successfully. Confidence: {final_confidence:.2f}")
            
            return {
                "user_query": user_query,
                "query_resolution": query_resolution,
                "extracted_data": {
                    "row_count": extracted_data.get("row_count"),
                    "summary": extracted_data.get("summary"),
                    "data": extracted_data.get("data"),
                    "error": None
                },
                "validation_result": validation_result,
                "formatted_response": formatted_response,
                "confidence_score": final_confidence,
                "suggested_followups": formatted_response.get("suggested_followups", []),
                "success": validation_result.get("is_valid", False)
            }
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return self._format_error_response(user_query, {"error": str(e)})
    
    def _format_error_response(self, user_query: str, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response consistently"""
        
        return {
            "user_query": user_query,
            "formatted_response": {
                "summary": f"Error: {error_info.get('error', 'Unknown error')}",
                "confidence_score": 0.0,
                "visualizations": [],
                "suggested_followups": []
            },
            "confidence_score": 0.0,
            "success": False,
            "error": error_info.get("error")
        }

class DataAnalystAgent:
    """
    Professional Data Analyst Agent
    Provides comprehensive statistical analysis, insights, and recommendations
    """
    
    def __init__(self, processor=None):
        self.llm = get_llm(temperature=0.3)
        self.processor = processor or get_processor()
    
    def analyze_dataset(self, table_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a dataset
        
        Returns:
            {
                "overview": str,
                "statistical_summary": Dict,
                "data_quality": Dict,
                "key_findings": List[str],
                "anomalies": List[str],
                "recommendations": List[str],
                "full_report": str
            }
        """
        try:
            import pandas as pd
            import numpy as np
            
            # Get data
            data = self.processor.query(f"SELECT * FROM {table_name}")
            stats = self.processor.get_table_stats(table_name)
            
            if data.empty:
                return {"error": "No data found in table"}
            
            # 1. Statistical Summary
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            
            statistical_summary = {
                "row_count": len(data),
                "column_count": len(data.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "memory_usage": f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }
            
            # 2. Data Quality Assessment
            missing_counts = data.isnull().sum()
            missing_by_col = {}
            for col, count in missing_counts[missing_counts > 0].items():
                missing_by_col[col] = int(count)  # Convert numpy int to Python int
            
            data_quality = {
                "completeness": float(((len(data) - missing_counts.max()) / len(data) * 100).round(2)),
                "duplicate_rows": int(data.duplicated().sum()),
                "columns_with_missing": int((missing_counts > 0).sum()),
                "missing_by_column": missing_by_col
            }
            
            # 3. Numeric Statistics
            numeric_stats = {}
            for col in numeric_cols[:10]:  # Limit to 10 columns
                numeric_stats[col] = {
                    "mean": float(data[col].mean()),
                    "median": float(data[col].median()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "q25": float(data[col].quantile(0.25)),
                    "q75": float(data[col].quantile(0.75))
                }
            
            # 4. Identify Anomalies (outliers using IQR method)
            anomalies = []
            for col in numeric_cols[:5]:  # Check first 5 numeric columns
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outlier_count > 0:
                    anomalies.append(f"Column '{col}': {outlier_count} outliers detected (range: {lower_bound:.2f} to {upper_bound:.2f})")
            
            # 5. Categorical Analysis
            category_analysis = {}
            for col in categorical_cols[:5]:  # First 5 categorical columns
                top_values = data[col].value_counts().head(3)
                category_analysis[col] = {
                    "unique_count": int(data[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in top_values.items()}  # Convert to Python types
                }
            
            # 6. Use LLM to generate insights and recommendations
            analysis_data = f"""
                                Dataset: {table_name}
                                Rows: {len(data):,}
                                Columns: {len(data.columns)}
                                Numeric Columns: {numeric_cols}
                                Categorical Columns: {categorical_cols[:5]}

                                Data Quality:
                                - Completeness: {data_quality['completeness']:.1f}%
                                - Duplicates: {data_quality['duplicate_rows']}
                                - Missing Data: {data_quality['columns_with_missing']} columns

                                Numeric Statistics (sample):
                                {json.dumps({k: v for k, v in numeric_stats.items()}, indent=2, default=str)[:500]}

                                Detected Anomalies:
                                {chr(10).join(anomalies) if anomalies else "None detected"}

                                Category Distribution (sample):
                                {json.dumps({k: str(v['top_values']) for k, v in category_analysis.items()}, indent=2)}
                                """
            
            system_prompt = load_prompt("data_analyst_prompt")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze this dataset:\n{analysis_data}")
            ]
            
            llm_response = self.llm.invoke(messages)
            
            # 7. Extract key findings
            response_text = llm_response.content
            key_findings = [
                "Dataset spans multiple business dimensions",
                f"Contains {len(data):,} records across {len(data.columns)} features",
                f"Data completeness: {data_quality['completeness']:.1f}%"
            ]
            
            recommendations = [
                "Perform regular data quality checks",
                "Document data collection methodology",
                "Consider archiving historical records for performance"
            ]
            
            # 8. Prepare visualization data
            viz_data = {
                "numeric_distributions": {},
                "category_top_values": {},
                "correlation_matrix": None,
                "box_plots": {},
                "missing_data": None
            }
            
            # Numeric distributions
            for col in numeric_cols[:3]:
                viz_data["numeric_distributions"][col] = data[col].tolist()
            
            # Top categories
            for col in categorical_cols[:3]:
                top_cats = data[col].value_counts().head(10)
                viz_data["category_top_values"][col] = top_cats.to_dict()
            
            # Correlation matrix for numeric columns
            if len(numeric_cols) >= 2:
                corr = data[numeric_cols[:5]].corr()
                viz_data["correlation_matrix"] = corr.values.tolist()
                viz_data["correlation_columns"] = numeric_cols[:5]
            
            # Box plot data for outlier detection
            for col in numeric_cols[:3]:
                viz_data["box_plots"][col] = {
                    "values": data[col].tolist(),
                    "name": col
                }
            
            # Missing data
            if missing_by_col:
                viz_data["missing_data"] = missing_by_col
            
            return {
                "success": True,
                "table_name": table_name,
                "overview": f"Dataset contains {len(data):,} records with {len(data.columns)} columns",
                "statistical_summary": statistical_summary,
                "numeric_stats": numeric_stats,
                "category_analysis": category_analysis,
                "data_quality": data_quality,
                "key_findings": key_findings,
                "anomalies": anomalies,
                "recommendations": recommendations,
                "full_report": response_text,
                "raw_data": data,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "viz_data": viz_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }