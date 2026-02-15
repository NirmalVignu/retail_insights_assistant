"""
Response Formatter for Q&A System
Formats agent responses with charts, tables, and confidence indicators
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from .data_processor import get_processor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Formats Q&A responses with visualizations and structured data
    """
    
    def __init__(self, processor=None):
        self.processor = processor or get_processor()
    
    def format_response(
        self,
        summary: str,
        data: Optional[pd.DataFrame] = None,
        query_result: Optional[Dict[str, Any]] = None,
        confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        Format response with summary, data, visualizations, and metadata
        
        Args:
            summary: Text answer
            data: Result dataframe
            query_result: Parsed query info
            confidence: Confidence score
            
        Returns:
            {
                "summary": "text answer",
                "confidence_score": 0.85,
                "data_preview": dataframe or None,
                "row_count": int,
                "visualizations": [chart specs],
                "suggested_followups": ["question1", "question2"],
                "metadata": {
                    "query_type": "analytical",
                    "tables_queried": ["table1"],
                    "processing_time_ms": 123
                }
            }
        """
        
        formatted = {
            "summary": summary,
            "confidence_score": confidence,
            "data_preview": None,
            "row_count": 0,
            "visualizations": [],
            "suggested_followups": [],
            "metadata": {
                "query_type": query_result.get("query_type", "unknown") if query_result else "unknown",
                "tables_queried": [query_result.get("primary_table")] if query_result else [],
                "processing_time_ms": 0
            }
        }
        
        # Add data preview
        if data is not None and len(data) > 0:
            formatted["row_count"] = len(data)
            formatted["data_preview"] = data.head(10).to_dict('records')
            
            # Generate visualizations based on query type
            visualizations = self._generate_visualizations(data, query_result)
            formatted["visualizations"] = visualizations
        
        # Add suggested follow-ups
        if query_result:
            formatted["suggested_followups"] = self._suggest_followups(query_result)
        
        return formatted
    
    def _generate_visualizations(
        self,
        data: pd.DataFrame,
        query_result: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate appropriate visualizations based on data and query type
        
        Args:
            data: Result dataframe
            query_result: Parsed query info
            
        Returns:
            List of chart specifications
        """
        visualizations = []
        
        try:
            if data is None or len(data) == 0:
                return visualizations
            
            query_type = query_result.get("query_type", "") if query_result else ""
            suggested = query_result.get("suggested_visualizations", []) if query_result else []
            
            # Get numeric and categorical columns
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Summary statistics table
            if len(numeric_cols) > 0:
                stats = data[numeric_cols].describe().round(2)
                visualizations.append({
                    "type": "table",
                    "title": "Summary Statistics",
                    "data": stats.reset_index().to_dict('records')
                })
            
            # Time series chart if date column exists
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols and len(numeric_cols) > 0:
                date_col = date_cols[0]
                value_col = numeric_cols[0]
                
                try:
                    df_sorted = data.sort_values(date_col)
                    fig = px.line(
                        df_sorted,
                        x=date_col,
                        y=value_col,
                        title="Trend Over Time",
                        markers=True
                    )
                    visualizations.append({
                        "type": "line_chart",
                        "title": "Trend Over Time",
                        "spec": fig.to_json()
                    })
                except Exception as e:
                    logger.debug(f"Could not create time series: {e}")
            
            # Bar chart for top categories
            if categorical_cols and len(numeric_cols) > 0:
                cat_col = categorical_cols[0]
                val_col = numeric_cols[0]
                
                try:
                    top_data = data.nlargest(10, val_col)[[cat_col, val_col]]
                    fig = px.bar(
                        top_data,
                        x=cat_col,
                        y=val_col,
                        title="Top 10 by Value",
                        color=val_col,
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(showlegend=False)
                    visualizations.append({
                        "type": "bar_chart",
                        "title": "Top 10 by Value",
                        "spec": fig.to_json()
                    })
                except Exception as e:
                    logger.debug(f"Could not create bar chart: {e}")
            
            # Distribution histogram
            if len(numeric_cols) >= 1:
                fig = px.histogram(
                    data,
                    x=numeric_cols[0],
                    nbins=30,
                    title=f"Distribution of {numeric_cols[0]}",
                    color_discrete_sequence=["#1f77b4"]
                )
                visualizations.append({
                    "type": "histogram",
                    "title": f"Distribution of {numeric_cols[0]}",
                    "spec": fig.to_json()
                })
            
            # Box plot for outliers
            if len(numeric_cols) >= 1 and len(categorical_cols) > 0:
                try:
                    fig = px.box(
                        data,
                        x=categorical_cols[0],
                        y=numeric_cols[0],
                        title=f"{numeric_cols[0]} by {categorical_cols[0]}"
                    )
                    visualizations.append({
                        "type": "box_plot",
                        "title": f"Distribution by {categorical_cols[0]}",
                        "spec": fig.to_json()
                    })
                except Exception as e:
                    logger.debug(f"Could not create box plot: {e}")
            
            # Correlation heatmap if multiple numeric columns
            if len(numeric_cols) > 2:
                corr = data[numeric_cols].corr()
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale="RdBu",
                        zmid=0
                    )
                )
                fig.update_layout(
                    title="Correlation Matrix",
                    xaxis_title="",
                    yaxis_title=""
                )
                visualizations.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "spec": fig.to_json()
                })
            
            logger.info(f"Generated {len(visualizations)} visualizations")
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _suggest_followups(self, query_result: Dict[str, Any]) -> List[str]:
        """
        Suggest follow-up questions based on query result
        
        Args:
            query_result: Parsed query info
            
        Returns:
            List of suggested questions
        """
        suggestions = []
        query_type = query_result.get("query_type", "")
        
        follow_up_map = {
            "summary": [
                "Can you break this down by region?",
                "What's the trend over time?",
                "How does this compare to last period?"
            ],
            "analytical": [
                "Which segment drove this result?",
                "What's the growth rate?",
                "Show me the distribution."
            ],
            "comparison": [
                "What are the key differences?",
                "Which performed better overall?",
                "What's driving the variance?"
            ],
            "timeseries": [
                "What's the forecast?",
                "Where are the anomalies?",
                "What caused the peaks/valleys?"
            ]
        }
        
        return follow_up_map.get(query_type, [
            "Can you provide more details?",
            "What's the business impact?",
            "How does this trend?"
        ])
    
    def format_table_for_display(self, data: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
        """
        Format dataframe for display in Streamlit
        
        Args:
            data: DataFrame to format
            max_rows: Maximum rows to display
            
        Returns:
            Display-ready dict with data and metadata
        """
        return {
            "data": data.head(max_rows).to_dict('records'),
            "columns": data.columns.tolist(),
            "total_rows": len(data),
            "displayed_rows": min(len(data), max_rows),
            "dtypes": data.dtypes.to_dict()
        }
    
    def format_confidence_indicator(self, confidence: float) -> Dict[str, Any]:
        """
        Format confidence indicator
        
        Args:
            confidence: Score 0-1
            
        Returns:
            Formatted confidence indicator
        """
        if confidence >= 0.8:
            level = "High"
            color = "green"
        elif confidence >= 0.6:
            level = "Medium"
            color = "yellow"
        else:
            level = "Low"
            color = "red"
        
        return {
            "score": round(confidence, 2),
            "level": level,
            "color": color,
            "message": f"{level} confidence ({confidence:.0%}) - " + 
                      ("Answer is well-understood" if level == "High" else
                       "Answer may need clarification" if level == "Medium" else
                       "Answer uncertain - please review")
        }


if __name__ == "__main__":
    # Example usage
    formatter = ResponseFormatter()
    
    # Create sample data
    sample_data = pd.DataFrame({
        "Region": ["North", "South", "East", "West"] * 3,
        "Month": pd.date_range("2024-01-01", periods=12, freq="M").repeat(4)[:12].tolist() + 
                pd.date_range("2024-01-01", periods=4, freq="M").tolist(),
        "Sales": [10000, 15000, 12000, 18000] * 3,
        "Growth": [0.05, 0.08, -0.02, 0.12] * 3
    })
    
    query_result = {
        "query_type": "analytical",
        "primary_table": "sales",
        "suggested_visualizations": ["bar_chart", "line_chart"]
    }
    
    response = formatter.format_response(
        summary="Sales data shows strong regional variation with North growing 5% YoY",
        data=sample_data,
        query_result=query_result,
        confidence=0.85
    )
    
    print(f"Summary: {response['summary']}")
    print(f"Confidence: {response['confidence_score']}")
    print(f"Visualizations: {len(response['visualizations'])}")
    print(f"Follow-ups: {response['suggested_followups']}")
