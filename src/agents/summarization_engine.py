"""
Summarization Engine
Generates automated business insights from sales data
"""

from ..utils.data_processor import get_processor
from ..utils.llm_config import get_llm
from ..utils.prompt_loader import format_prompt, load_prompt
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from typing import Dict, List, Any

class SummarizationEngine:
    """
    Generates comprehensive summaries of sales data using LLM analysis
    """
    
    def __init__(self, processor=None):
        self.llm = get_llm(temperature=0.2)
        self.processor = processor or get_processor()
    
    def generate_summary(self, table_name: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive summary of sales data
        
        Returns:
            {
                "executive_summary": str,
                "key_metrics": Dict,
                "insights": List[str],
                "recommendations": List[str],
                "data_quality": Dict
            }
        """
        
        # Get first table if not specified
        if not table_name:
            tables = self.processor.list_tables()
            if not tables:
                return {"error": "No data tables available"}
            table_name = tables[0]
        
        try:
            # Get statistics
            stats = self.processor.get_table_stats(table_name)
            
            # Get sample data
            sample_data = self.processor.query(f"SELECT * FROM {table_name} LIMIT 10")
            
            # Calculate numeric insights
            numeric_cols = self.processor.get_numeric_columns(table_name)
            numeric_insights = {}
            
            for col in numeric_cols[:5]:  # Limit to 5 numeric columns
                try:
                    result = self.processor.query(
                        f"SELECT MIN({col}) as min, MAX({col}) as max, AVG({col}) as avg FROM {table_name}"
                    )
                    numeric_insights[col] = {
                        "min": float(result.iloc[0]['min']) if pd.notna(result.iloc[0]['min']) else 0,
                        "max": float(result.iloc[0]['max']) if pd.notna(result.iloc[0]['max']) else 0,
                        "avg": float(result.iloc[0]['avg']) if pd.notna(result.iloc[0]['avg']) else 0,
                    }
                except:
                    pass
            
            # Build context for LLM
            data_context = f"""
Table: {table_name}
Total Records: {stats.get('row_count', 'N/A')}
Columns: {stats.get('column_count', 'N/A')}

Numeric Metrics:
{json.dumps(numeric_insights, indent=2)}

Sample Data:
{sample_data.head().to_string()}
"""
            
            # Generate summary using LLM
            system_prompt = load_prompt("summarization_prompt")            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze this retail sales data in detail:\n\n{data_context}\n\nProvide data-driven insights using the exact structure specified.")
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                "table_name": table_name,
                "summary": response.content,
                "statistics": stats,
                "numeric_insights": numeric_insights,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Summary generation failed: {str(e)}",
                "success": False
            }
    
    def generate_comparative_summary(self, table1: str, table2: str) -> Dict[str, Any]:
        """
        Compare two tables and generate comparative insights
        """
        
        try:
            stats1 = self.processor.get_table_stats(table1)
            stats2 = self.processor.get_table_stats(table2)
            
            comparison_context = f"""
Table 1: {table1}
- Rows: {stats1.get('row_count', 'N/A')}
- Columns: {stats1.get('column_count', 'N/A')}

Table 2: {table2}
- Rows: {stats2.get('row_count', 'N/A')}
- Columns: {stats2.get('column_count', 'N/A')}
"""
            
            system_prompt = load_prompt("comparative_summarization_prompt")
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"{comparison_context}\n\nProvide comprehensive, quantified comparative analysis with integration roadmap.")
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                "comparison": response.content,
                "table1": table1,
                "table2": table2,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Comparative summary failed: {str(e)}",
                "success": False
            }


# For imports
import json
