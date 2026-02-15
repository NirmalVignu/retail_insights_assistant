"""
Streamlit Dashboard for Retail Insights Assistant - Restructured
Interactive UI for data summarization and Q&A
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.data_processor import get_processor
from src.agents.multi_agent import MultiAgentOrchestrator, DataAnalystAgent
from src.agents.summarization_engine import SummarizationEngine

# Optional imports - may not be available
try:
    from src.utils.pdf_report_generator import generate_pdf_report
    _has_pdf_export = True
except ImportError:
    generate_pdf_report = None
    _has_pdf_export = False

try:
    from src.utils.conversation_memory import ConversationMemory
    _has_memory = True
except ImportError:
    ConversationMemory = None
    _has_memory = False

from src.utils.response_formatter import ResponseFormatter
from src.utils.config import PAGE_TITLE, PAGE_LAYOUT, INITIAL_SIDEBAR_STATE, CSV_DATA_PATH
from src.utils.input_loader import infer_table_name
from src.graph import LangGraphAgent
import plotly.express as px
import plotly.graph_objects as go
import time
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📊",
    layout=PAGE_LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# Performance optimization: Cache slow operations
@st.cache_resource
def get_cached_processor():
    """Cache the processor instance"""
    return get_processor()

@st.cache_resource
def get_cached_conversation_memory():
    """Cache conversation memory initialization"""
    if ConversationMemory is not None:
        return ConversationMemory()
    return None

@st.cache_resource
def load_default_data(_processor):
    """Cache the default CSV data loading"""
    try:
        result = _processor.load_all_csvs(CSV_DATA_PATH)
        tables = _processor.list_tables()
        logger.info(f"Default data loaded: {len(tables)} tables")
        return True
    except Exception as e:
        logger.error(f"Error loading default data: {e}")
        return False

# Visualization helper functions
def create_numeric_distribution(data: pd.DataFrame, column: str):
    """Create histogram for numeric column"""
    fig = px.histogram(
        data, 
        x=column,
        title=f"Distribution of {column}",
        labels={column: column, "count": "Frequency"},
        nbins=30,
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(showlegend=False, height=400)
    return fig

def create_top_categories(data: pd.DataFrame, column: str, top_n: int = 10):
    """Create bar chart for top categories"""
    value_counts = data[column].value_counts().head(top_n)
    
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title=f"Top {top_n} {column} Values",
        labels={"x": column, "y": "Count"},
        color_discrete_sequence=['#764ba2']
    )
    fig.update_layout(showlegend=False, height=400)
    return fig

def create_category_chart(data: pd.DataFrame, column: str, top_n: int = 10):
    """Create bar chart for categorical column"""
    value_counts = data[column].value_counts().head(top_n)
    
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title=f"Top {top_n} {column} Values",
        labels={"x": column, "y": "Count"}
    )
    fig.update_layout(showlegend=False)
    return fig

def create_correlation_heatmap(data: pd.DataFrame):
    """Create correlation heatmap for numeric columns"""
    numeric_cols = data.select_dtypes(include=['number']).columns # Limit to first 8 columns
    if len(numeric_cols) > 0:
        corr_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title="Correlation Matrix")
        return fig
    return None

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    layout=PAGE_LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# Custom CSS styling
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 16px; font-weight: 500; }
    .metric-box { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white; 
        padding: 1.5rem; 
        border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)



# Initialize session state
if "processor" not in st.session_state:
    st.session_state.processor = get_cached_processor()

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = get_cached_conversation_memory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_tables" not in st.session_state:
    st.session_state.uploaded_tables = []

# Sidebar
st.sidebar.title("📁 Data Management")

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV/Excel files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Upload your sales data files for analysis"
)

# Process uploaded files
if uploaded_files:
    processor = st.session_state.processor
    
    for uploaded_file in uploaded_files:
        # Create a unique identifier for this file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if file_id not in st.session_state.uploaded_tables:
            with st.sidebar:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Read file based on extension
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        # Infer table name
                        table_name = infer_table_name(uploaded_file.name)
                        
                        # Load into DuckDB
                        processor.load_dataframe(df, table_name)
                        
                        # Track uploaded table
                        st.session_state.uploaded_tables.append(file_id)
                        
                        st.sidebar.success(f"✅ Loaded: {table_name} ({len(df):,} rows)")
                        
                    except Exception as e:
                        st.sidebar.error(f"❌ Error loading {uploaded_file.name}: {e}")

# Load default data if no files uploaded
if not st.session_state.uploaded_tables:
    load_default_data(st.session_state.processor)

# Display loaded tables
tables = st.session_state.processor.list_tables()
if tables:
    st.sidebar.success(f"📊 {len(tables)} table(s) loaded")
    with st.sidebar.expander("View Tables"):
        for table in tables:
            st.write(f"- {table}")
else:
    st.sidebar.warning("⚠️ No data loaded")

# Main Content
st.title("🛍️ Retail Insights Assistant")
st.markdown("### Multi-Agent AI System for Sales Analytics")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Summarization",
    "💬 Q&A Chat",
    "🔍 Data Explorer",
    "📈 Data Analyst"
])

# TAB 1: Summarization
with tab1:
    st.header("📊 Automated Data Summarization")
    st.markdown("Generate comprehensive business intelligence reports from your data.")
    
    if not tables:
        st.warning("⚠️ Please upload data files to begin.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_table = st.selectbox("Select table to summarize:", tables)
        
        with col2:
            summarize_btn = st.button("🚀 Generate Summary", type="primary", use_container_width=True)
        
        # Comparative analysis option
        if len(tables) >= 2:
            st.divider()
            st.subheader("🔄 Comparative Analysis")
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                table1 = st.selectbox("Table 1:", tables, key="comp_table1")
            with col2:
                table2 = st.selectbox("Table 2:", [t for t in tables if t != table1], key="comp_table2")
            with col3:
                compare_btn = st.button("⚡ Compare", use_container_width=True)
        
        # Generate summary
        if summarize_btn:
            with st.spinner("🔄 Generating comprehensive summary..."):
                engine = SummarizationEngine(st.session_state.processor)
                summary_result = engine.generate_summary(selected_table)
                
                if summary_result.get("success"):
                    st.success("✅ Summary generated successfully!")
                    
                    # Display summary
                    st.markdown(summary_result["summary"])
                    
                    # Display statistics
                    with st.expander("📊 Detailed Statistics"):
                        stats = summary_result.get("statistics", {})
                        if stats:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Rows", f"{stats.get('row_count', 0):,}")
                            with col2:
                                st.metric("Columns", stats.get('column_count', 0))
                            with col3:
                                st.metric("Complete %", f"{stats.get('completeness', 0):.1f}%")
                            with col4:
                                st.metric("Duplicates", stats.get('duplicate_count', 0))
                    
                    # Export option
                    st.divider()
                    if _has_pdf_export and st.button("📄 Export as PDF"):
                        with st.spinner("Generating PDF..."):
                            pdf_path = generate_pdf_report(
                                table_name=selected_table,
                                summary_text=summary_result["summary"],
                                statistics=stats
                            )
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download PDF Report",
                                    f.read(),
                                    file_name=f"{selected_table}_summary.pdf",
                                    mime="application/pdf"
                                )
                    elif not _has_pdf_export:
                        st.info("💡 Install reportlab to enable PDF export: `pip install reportlab`")
                else:
                    st.error(f"❌ Error: {summary_result.get('error', 'Unknown error')}")
        
        # Comparative analysis
        if len(tables) >= 2 and 'compare_btn' in locals() and compare_btn:
            with st.spinner("🔄 Performing comparative analysis..."):
                engine = SummarizationEngine(st.session_state.processor)
                comparison_result = engine.generate_comparative_summary(table1, table2)
                
                if comparison_result.get("success"):
                    st.success("✅ Comparison completed!")
                    st.markdown(comparison_result["comparison"])
                else:
                    st.error(f"❌ Error: {comparison_result.get('error', 'Unknown error')}")

# TAB 2: Q&A Chat
with tab2:
    st.header("💬 Interactive Q&A Chat")
    st.markdown("Ask natural language questions about your data.")
    
    if not tables:
        st.warning("⚠️ Please upload data files to begin chatting.")
    else:
        # Data source selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_source = st.selectbox(
                "📊 Select Data Source:",
                ["All Tables"] + tables,
                help="Choose a specific table or query across all tables"
            )
        with col2:
            st.metric("Available Tables", len(tables))
        
        st.divider()
        
        # Agent selection
        col1, col2 = st.columns([3, 1])
        with col1:
            agent_type = st.radio(
                "Select Agent:",
                ["🧠 LangGraph (Advanced)", "🤖 Multi-Agent (Fast)"],
                horizontal=True
            )
        with col2:
            clear_btn = st.button("🗑️ Clear Chat", use_container_width=True)
            if clear_btn:
                st.session_state.chat_history = []
                st.rerun()
        
        # Display selected source info
        if selected_source != "All Tables":
            st.info(f"🎯 Querying: **{selected_source}**")
        
        # Example queries
        with st.expander("💡 Example Questions", expanded=False):
            if selected_source == "All Tables":
                st.markdown("""
                - What are the total sales across all datasets?
                - Compare revenue trends between different months
                - Which categories have the highest performance?
                - Show me the distribution of orders by region
                """)
            else:
                st.markdown(f"""
                - What is the total revenue in {selected_source}?
                - Show top 10 products by sales
                - What are the sales trends over time?
                - Which region has the highest orders?
                - Give me a summary of {selected_source}
                """)
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    # Show source badge if available
                    if "source" in msg and msg["source"] != "All Tables":
                        st.caption(f"📊 Source: {msg['source']}")
                    
                    st.markdown(msg["content"])
                    
                    # Display chart if available
                    if "chart" in msg and msg["chart"] is not None:
                        st.plotly_chart(msg["chart"], use_container_width=True)
                    
                    # Display confidence score
                    if "confidence" in msg and msg["role"] == "assistant":
                        confidence = msg["confidence"]
                        if confidence >= 0.8:
                            st.success(f"Confidence: {confidence:.0%} ✓")
                        elif confidence >= 0.6:
                            st.warning(f"Confidence: {confidence:.0%} ⚠")
                        else:
                            st.error(f"Low confidence: {confidence:.0%} ⚠️")
        
        # Chat input with dynamic placeholder
        if selected_source == "All Tables":
            placeholder = "Ask me anything about your data..."
        else:
            placeholder = f"Ask about {selected_source}..."
        
        user_query = st.chat_input(placeholder)
        
        if user_query:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query,
                "source": selected_source
            })
            
            # Process query
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Enhance query with table context if specific table selected
                        enhanced_query = user_query
                        if selected_source != "All Tables":
                            enhanced_query = f"From the '{selected_source}' table: {user_query}"
                        
                        if "LangGraph" in agent_type:
                            # Use LangGraph agent
                            agent = LangGraphAgent(
                                st.session_state.processor,
                                st.session_state.conversation_memory
                            )
                            result = agent.process_query(enhanced_query)
                        else:
                            # Use Multi-Agent orchestrator
                            orchestrator = MultiAgentOrchestrator(
                                st.session_state.processor,
                                st.session_state.conversation_memory,
                                ResponseFormatter(st.session_state.processor)
                            )
                            result = orchestrator.process_query(enhanced_query)
                        
                        # Display response
                        response_text = result.get("formatted_response", {}).get("summary", "No response generated")
                        st.markdown(response_text)
                        
                        # Display chart if available
                        chart = result.get("formatted_response", {}).get("primary_chart")
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Display confidence
                        confidence = result.get("formatted_response", {}).get("confidence_score", 0.0)
                        if confidence >= 0.8:
                            st.success(f"Confidence: {confidence:.0%} ✓")
                        elif confidence >= 0.6:
                            st.warning(f"Confidence: {confidence:.0%} ⚠")
                        else:
                            st.error(f"Low confidence: {confidence:.0%} ⚠️")
                        
                        # Save to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_text,
                            "chart": chart,
                            "confidence": confidence,
                            "source": selected_source
                        })
                        
                        # Save to conversation memory (if available)
                        if st.session_state.conversation_memory is not None:
                            st.session_state.conversation_memory.add_message(
                                role="user",
                                content=user_query,
                                metadata={
                                    "timestamp": "query",
                                    "selected_source": selected_source
                                }
                            )
                            st.session_state.conversation_memory.add_message(
                                role="assistant",
                                content=response_text,
                                metadata={
                                    "confidence": confidence,
                                    "source": selected_source
                                }
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"Query processing error: {e}", exc_info=True)

# TAB 3: Data Explorer
with tab3:
    st.header("🔍 Interactive Data Explorer")
    st.markdown("Explore and visualize your data interactively.")
    
    if not tables:
        st.warning("⚠️ Please upload data files to explore.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_table = st.selectbox("Select table:", tables, key="explorer_table")
        with col2:
            st.write("")  # Spacing
            show_all = st.checkbox("Show all rows", key="show_all_explorer")
        
        if selected_table:
            # Create sub-tabs within Data Explorer
            explorer_tab1, explorer_tab2, explorer_tab3 = st.tabs(["📋 Data View", "📊 Visualizations", "📈 Analytics"])
            
            with explorer_tab1:
                st.subheader("Table Data")
                
                if st.button("📥 Load Data", key="explorer_load_btn", use_container_width=True):
                    try:
                        processor = st.session_state.processor
                        
                        if show_all:
                            data = processor.query(f"SELECT * FROM {selected_table}")
                        else:
                            data = processor.query(f"SELECT * FROM {selected_table} LIMIT 100")
                        
                        st.dataframe(data, use_container_width=True)
                        
                        # Table statistics
                        st.subheader("📊 Table Statistics")
                        stats = processor.get_table_stats(selected_table)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", f"{stats.get('row_count', 0):,}")
                        with col2:
                            st.metric("Columns", stats.get('column_count', 0))
                        with col3:
                            st.metric("Displayed Rows", f"{len(data):,}")
                        
                        # Column information
                        st.subheader("🏷️ Column Information")
                        schema = stats.get("columns", {})
                        schema_df = pd.DataFrame(list(schema.items()), columns=["Column", "Type"])
                        st.dataframe(schema_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error loading data: {str(e)}")
            
            with explorer_tab2:
                st.subheader("Visual Analysis")
                
                if st.button("📊 Load Visualizations", key="explorer_viz_btn", use_container_width=True):
                    try:
                        processor = st.session_state.processor
                        # Get sample data for visualization (limit for performance)
                        sample_data = processor.query(f"SELECT * FROM {selected_table} LIMIT 5000")
                        
                        if sample_data is not None and not sample_data.empty:
                            numeric_cols = sample_data.select_dtypes(include=['number']).columns.tolist()
                            categorical_cols = sample_data.select_dtypes(include=['object']).columns.tolist()
                            
                            # Numeric distributions
                            if numeric_cols:
                                st.write("#### 📈 Numeric Distributions")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    selected_numeric = st.selectbox("Numeric column:", numeric_cols, key="exp_numeric")
                                    fig = create_numeric_distribution(sample_data, selected_numeric)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    if len(numeric_cols) > 1:
                                        second_numeric = st.selectbox("Second numeric column:", numeric_cols,
                                                                     index=1 if len(numeric_cols) > 1 else 0,
                                                                     key="exp_numeric2")
                                        fig = create_numeric_distribution(sample_data, second_numeric)
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # Category analysis
                            if categorical_cols:
                                st.write("#### 🏷️ Category Analysis")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    selected_cat = st.selectbox("Category column:", categorical_cols, key="exp_cat")
                                    fig = create_top_categories(sample_data, selected_cat, top_n=15)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.write("#### Summary by Category")
                                    if numeric_cols:
                                        selected_agg = st.selectbox("Aggregate by:", numeric_cols, key="exp_agg")
                                        grouped = sample_data.groupby(selected_cat)[selected_agg].agg(['count', 'mean', 'sum'])
                                        st.dataframe(grouped.head(10), use_container_width=True)
                            
                            # Correlation heatmap
                            if len(numeric_cols) >= 2:
                                st.write("#### 🔗 Correlation Matrix")
                                corr_matrix = sample_data[numeric_cols].corr()
                                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                               color_continuous_scale='RdBu',
                                               title="Correlation Between Numeric Columns")
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No data available for visualization")
                    except Exception as e:
                        st.warning(f"⚠️ Visualization error: {str(e)}")
            
            with explorer_tab3:
                st.subheader("Advanced Analytics")
                
                if st.button("📈 Load Analytics", key="explorer_analytics_btn", use_container_width=True):
                    try:
                        processor = st.session_state.processor
                        data = processor.query(f"SELECT * FROM {selected_table} LIMIT 10000")
                        
                        if data is not None and not data.empty:
                            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                            
                            # Descriptive statistics
                            st.write("#### 📊 Descriptive Statistics")
                            if numeric_cols:
                                desc_stats = data[numeric_cols].describe().T
                                st.dataframe(desc_stats, use_container_width=True)
                            
                            # Missing data analysis
                            st.write("#### 🔍 Missing Data Analysis")
                            missing_data = pd.DataFrame({
                                'Column': data.columns,
                                'Missing Count': data.isnull().sum(),
                                'Missing %': (data.isnull().sum() / len(data) * 100).round(2)
                            })
                            missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                            
                            if len(missing_data) > 0:
                                st.dataframe(missing_data, use_container_width=True)
                                
                                # Visualize missing data
                                fig = px.bar(missing_data, x='Column', y='Missing %',
                                           title="Missing Data by Column (%)",
                                           color_discrete_sequence=['#ff6b6b'])
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.success("✅ No missing data found!")
                            
                            # Data types summary
                            st.write("#### 📋 Data Types Summary")
                            dtype_counts = data.dtypes.value_counts().reset_index()
                            dtype_counts.columns = ['Data Type', 'Count']
                            st.dataframe(dtype_counts, use_container_width=True)
                            
                            # Correlation heatmap
                            if len(numeric_cols) >= 2:
                                st.write("#### 🔗 Correlation Analysis")
                                fig = create_correlation_heatmap(data)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No data available for analytics")
                    except Exception as e:
                        st.error(f"❌ Analytics error: {str(e)}")

# TAB 4: Data Analyst
with tab4:
    st.header("📈 Comprehensive Data Analysis")
    st.markdown("Get expert-level dataset profiling and insights.")
    
    if not tables:
        st.warning("⚠️ Please upload data files to analyze.")
    else:
        selected_table = st.selectbox("Select table for analysis:", tables, key="analyst_table")
        
        analyze_btn = st.button("🔬 Perform Deep Analysis", type="primary", use_container_width=True)
        
        if analyze_btn and selected_table:
            with st.spinner("🔄 Performing comprehensive analysis..."):
                try:
                    analyst = DataAnalystAgent(st.session_state.processor)
                    analysis_result = analyst.analyze_dataset(selected_table)
                    
                    if analysis_result.get("success"):
                        st.success("✅ Analysis completed!")
                        
                        # Display comprehensive report
                        st.markdown(analysis_result.get("full_report", "No report generated"))
                        
                        # Key metrics
                        st.divider()
                        col1, col2, col3, col4 = st.columns(4)
                        stats = analysis_result.get("statistical_summary", {})
                        data_quality = analysis_result.get("data_quality", {})
                        with col1:
                            st.metric("Rows", f"{stats.get('row_count', 0):,}")
                        with col2:
                            st.metric("Columns", stats.get('column_count', 0))
                        with col3:
                            st.metric("Completeness", f"{data_quality.get('completeness', 0):.1f}%")
                        with col4:
                            duplicates = data_quality.get('duplicate_rows', 0)
                            st.metric("Duplicates", f"{duplicates:,}")
                        
                        # Get visualization data
                        viz_data = analysis_result.get("viz_data", {})
                        raw_data = analysis_result.get("raw_data")
                        numeric_cols = analysis_result.get("numeric_cols", [])
                        categorical_cols = analysis_result.get("categorical_cols", [])
                        
                        # Visualizations with sub-tabs
                        if viz_data:
                            st.divider()
                            st.subheader("📊 Analytical Visualizations")
                            
                            # Create visualization sub-tabs
                            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
                                ["📈 Distributions", "🏷️ Categories", "🔗 Correlation", "📦 Box Plots"]
                            )
                            
                            # Tab 1: Numeric Distributions
                            with viz_tab1:
                                st.write("#### Numeric Column Distributions")
                                if numeric_cols and raw_data is not None:
                                    viz_cols = st.columns(min(2, len(numeric_cols[:3])))
                                    for idx, col in enumerate(numeric_cols[:3]):
                                        with viz_cols[idx % 2]:
                                            fig = px.histogram(raw_data, x=col, nbins=30,
                                                              title=f"Distribution of {col}",
                                                              color_discrete_sequence=['#667eea'])
                                            fig.update_layout(height=350, showlegend=False)
                                            st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No numeric columns available")
                            
                            # Tab 2: Category Analysis
                            with viz_tab2:
                                st.write("#### Top Values in Categories")
                                if category_analysis := analysis_result.get('category_analysis', {}):
                                    for col_name, data_info in list(category_analysis.items())[:2]:
                                        top_values = data_info.get('top_values', {})
                                        if top_values:
                                            fig = px.bar(x=list(top_values.values()),
                                                       y=list(top_values.keys()),
                                                       title=f"Top Values in {col_name}",
                                                       orientation='h',
                                                       color_discrete_sequence=['#764ba2'])
                                            fig.update_layout(height=350, showlegend=False)
                                            st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No categorical columns available")
                            
                            # Tab 3: Correlation Heatmap
                            with viz_tab3:
                                st.write("#### Correlation Matrix")
                                if numeric_cols and len(numeric_cols) >= 2 and raw_data is not None:
                                    corr_data = raw_data[numeric_cols[:5]].corr()
                                    fig = px.imshow(corr_data, text_auto=True, aspect="auto",
                                                   color_continuous_scale='RdBu',
                                                   title="Correlation Between Numeric Columns",
                                                   labels=dict(x="Features", y="Features", color="Correlation"))
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Need at least 2 numeric columns for correlation analysis")
                            
                            # Tab 4: Box Plots
                            with viz_tab4:
                                st.write("#### Box Plots (Outlier Detection)")
                                if numeric_cols and raw_data is not None:
                                    # Create box plot for multiple columns
                                    box_data = raw_data[numeric_cols[:5]].melt(var_name='Column', value_name='Value')
                                    fig = px.box(box_data, x='Column', y='Value',
                                                title="Box Plots for Numeric Columns",
                                                color_discrete_sequence=['#667eea'])
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No numeric columns available for box plots")
                    
                    else:
                        st.error(f"❌ Error: {analysis_result.get('error', 'Analysis failed')}")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {e}", exc_info=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🛍️ Retail Insights Assistant v1.0 | Multi-Agent AI System with LangGraph</p>
    <p>Powered by LangChain, DuckDB, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
