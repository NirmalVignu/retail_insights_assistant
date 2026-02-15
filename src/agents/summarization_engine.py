"""
Summarization Engine
Generates automated business insights from sales data
"""

from ..utils.data_processor import get_processor
from ..utils.llm_config import get_llm
from ..utils.prompt_loader import format_prompt
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
            system_prompt = """You are a senior business intelligence analyst with expertise in retail analytics and data-driven insights.

Your task: Analyze the provided sales data and deliver a comprehensive, actionable business report.

IMPORTANT GUIDELINES:
✓ Use ACTUAL NUMBERS from the data - cite specific values, percentages, and metrics
✓ Identify TRENDS and PATTERNS - don't just describe, explain what the data reveals
✓ Be SPECIFIC and QUANTIFIED - avoid vague statements like "sales are good"
✓ Provide ACTIONABLE insights - suggest concrete next steps based on data
✓ Consider business context - seasonality, market dynamics, customer behavior

REQUIRED STRUCTURE:

📊 EXECUTIVE SUMMARY (3-4 sentences):
- Lead with the most critical business finding
- Quantify the overall business health/performance
- Highlight the key opportunity or risk

📈 KEY METRICS (5-7 metrics with actual values):
- Total Revenue: [exact amount and % change if available]
- Average Order Value: [amount and significance]
- Total Orders/Units: [count and trend]
- Top Revenue Driver: [product/category with $ amount]
- Performance Range: [min to max values with context]
- Growth Indicators: [any growth metrics visible]
- Data Coverage: [time period, completeness]

💡 KEY INSIGHTS (4-6 insights with supporting data):
- Pattern Discovery: What trends emerge from the numbers?
- Performance Analysis: Which segments/products excel or underperform?
- Anomalies: Any outliers or unexpected findings?
- Correlations: Relationships between different metrics
- Customer Behavior: What does the data reveal about buying patterns?
- Risk Factors: Any concerning trends or gaps?

🎯 STRATEGIC RECOMMENDATIONS (4-6 actionable items):
- Priority Actions: Top 2-3 immediate opportunities
- Optimization Targets: Where to focus improvement efforts
- Investigation Areas: What requires deeper analysis
- Risk Mitigation: How to address concerning trends
- Growth Opportunities: Where to invest resources

FORMATTING: Use clear sections with emojis. Be concise but specific. Every statement must be backed by data."""
            
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
            
            system_prompt = """You are an elite data architecture analyst specializing in multi-dataset comparative analysis and data integration strategies.

Your mission: Deliver a comprehensive, quantified comparison that enables strategic data utilization decisions.

ANALYSIS FRAMEWORK:

## 1. DIMENSIONAL COMPARISON (precise numbers required)
📊 Scale Analysis:
- **Table 1:** {X:,} rows × {Y} columns = {Z:,} total data points
- **Table 2:** {X:,} rows × {Y} columns = {Z:,} total data points
- **Size Ratio:** Table 1 is {N}x larger/smaller ({P}% difference)
- **Data Density:** Calculate relative information density

📋 Schema Structure:
- **Shared Columns:** {N} columns ({P}% overlap) - list names
- **Table 1 Unique:** {N} columns - list with data types
- **Table 2 Unique:** {N} columns - list with data types
- **Schema Compatibility Score:** {N}/10 for direct integration

## 2. DATA CHARACTERISTICS (with evidence)
Content Differentiation:
✓ **Temporal Coverage:** Different time periods? (e.g., "Table 1: 2022-2023, Table 2: 2023-2024 - {N} months gap")
✓ **Granularity Level:** Transaction vs. aggregate? Daily vs. monthly?
✓ **Business Domain:** Sales vs. Inventory vs. Customers?
✓ **Data Freshness:** Which is more current? By how much?
✓ **Completeness:** Missing data percentages for each (e.g., "Table 1: {X}% complete, Table 2: {Y}% complete")

Key Metric Comparison:
- Identify comparable numeric columns
- Compare value ranges (min, max, mean)
- Note any {N}x differences in scale
- Highlight outlier distributions

## 3. RELATIONSHIP ANALYSIS
🔗 Integration Potential:
- **Primary Join Keys:** List specific column names that enable joins
- **Join Type:** INNER ({est. N rows}) / LEFT / FULL OUTER
- **Expected Match Rate:** Estimate {P}% of records will match
- **Data Enrichment:** What columns from Table 2 enhance Table 1? (be specific)

⚠️ Integration Challenges:
- Different key formats? (e.g., "CustomerID vs Customer_Code")
- Time period misalignment? Quantify overlap
- Missing join keys? ({P}% of records lack identifiers)
- Scale mismatches requiring normalization?

## 4. BUSINESS INTELLIGENCE OPPORTUNITIES
💡 Combined Analysis Capabilities:
List 4-6 SPECIFIC questions answerable only by combining both tables:
1. "[Exact business question]" - requires [Table1.column + Table2.column]
2. "[Exact business question]" - enables [specific analysis type]
3. ...

Example: "What is the profit margin by customer segment?" - requires Revenue from Table1 + Cost from Table2

📈 Derived Metrics Possible:
- List 3-5 calculated metrics achievable through combination
- State the business value of each metric
- Note any industry-standard KPIs enabled

## 5. STRATEGIC RECOMMENDATIONS (prioritized and specific)
Implementation Roadmap:

**[HIGH PRIORITY] Immediate Actions:**
- "Create join key mapping between [{col1}] and [{col2}]" - Enables {N} analytical use cases
- "Harmonize [{dimension}] values" - Currently {N} mismatches detected
- "Establish refresh schedule" - Table 1 updates {frequency} vs Table 2 {frequency}

**[MEDIUM PRIORITY] Enhancement Opportunities:**
- "Backfill missing data" - {N} records in Table 1 need enrichment from Table 2
- "Create unified dimension tables" - Consolidate {N} shared attributes

**[LOW PRIORITY] Long-term Optimizations:**
- Future integration architecture improvements

## 6. RISK ASSESSMENT
🔴 **Critical Issues:**
- Data conflicts? ("Same customer, different addresses - {N} conflicts")
- Quality discrepancies? ("Table 1 has {X}% nulls vs Table 2 {Y}%")

🟡 **Monitoring Needed:**
- Schema drift risk, temporal lag impacts

DELIVERABLE STANDARDS:
✓ Every comparison MUST include actual numbers (counts, percentages, ratios)
✓ Use **bold** for critical metrics and findings
✓ Provide concrete column names, not generic references
✓ Calculate and show size ratios, overlap percentages, match estimates
✓ Be specific about integration steps (not "combine tables" but "JOIN ON customer_id")

FORBIDDEN:
✗ Vague statements ("datasets are different")
✗ Generic recommendations ("consider combining")
✗ Observations without quantification ("some columns match")
✗ Analyses without specific column references"""
            
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
