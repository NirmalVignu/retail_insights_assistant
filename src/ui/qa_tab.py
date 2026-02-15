"""
Enhanced Q&A Tab Implementation
Integrates conversation memory, confidence scoring, and visualizations
"""

import streamlit as st
import pandas as pd
from typing import Optional
try:
    from ..graph.langgraph_agent import LangGraphQueryAgent
    USE_LANGGRAPH = True
except ImportError:
    from ..agents.multi_agent import MultiAgentOrchestrator
    USE_LANGGRAPH = False
from ..utils.conversation_memory import ConversationMemory
import plotly.graph_objects as go
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def initialize_qa_session():
    """Initialize Q&A session state"""
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory()
    
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    
    if "qa_orchestrator" not in st.session_state:
        from data_processor import get_processor
        processor = st.session_state.get("processor")
        if processor is None:
            processor = get_processor()
        
        # Use LangGraph if available, otherwise fall back to multi-agent
        if USE_LANGGRAPH:
            logger.info("Using LangGraph-based agent (supports complex queries)")
            st.session_state.qa_orchestrator = LangGraphQueryAgent(
                processor=processor,
                conversation_memory=st.session_state.conversation_memory
            )
            st.session_state.use_langgraph = True
        else:
            logger.info("Using basic multi-agent orchestrator")
            st.session_state.qa_orchestrator = MultiAgentOrchestrator(
                processor=processor,
                conversation_memory=st.session_state.conversation_memory
            )
            st.session_state.use_langgraph = False


def display_confidence_indicator(confidence: float):
    """Display confidence score with visual indicator"""
    
    if confidence >= 0.85:
        level = "🟢 High Confidence"
        color = "#2ecc71"
        message = "Answer is well-understood and reliable"
    elif confidence >= 0.65:
        level = "🟡 Medium Confidence"
        color = "#f39c12"
        message = "Answer is reasonable but may need verification"
    else:
        level = "🔴 Low Confidence"
        color = "#e74c3c"
        message = "Answer may be inaccurate, please review or rephrase query"
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"**{level}**")
    with col2:
        st.progress(min(confidence, 1.0))
        st.caption(message)


def display_data_table(data: Optional[pd.DataFrame], max_rows: int = 10):
    """Display data table with formatting"""
    if data is None or data.empty:
        st.info("No data to display")
        return
    
    st.subheader(f"Data Summary ({len(data)} rows)")
    
    # Display key statistics
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(data))
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            if len(numeric_cols) > 0:
                st.metric("Numeric Columns", len(numeric_cols))
    
    # Display table
    st.dataframe(
        data.head(max_rows),
        use_container_width=True,
        height=400
    )
    
    if len(data) > max_rows:
        st.caption(f"Showing {max_rows} of {len(data)} rows")


def display_visualizations(visualizations: list):
    """Display generated visualizations"""
    if not visualizations:
        return
    
    st.subheader("📊 Visualizations")
    
    # Display visualizations in a grid
    cols = st.columns(2)
    for idx, viz in enumerate(visualizations[:4]):  # Limit to 4 charts
        with cols[idx % 2]:
            try:
                viz_type = viz.get("type", "unknown")
                title = viz.get("title", f"Chart {idx+1}")
                
                st.markdown(f"**{title}**")
                
                if viz_type == "table":
                    # Display table data
                    viz_df = pd.DataFrame(viz.get("data", []))
                    st.dataframe(viz_df, use_container_width=True)
                
                elif viz_type in ["line_chart", "bar_chart", "histogram", "box_plot", "heatmap"]:
                    # Display Plotly chart
                    import plotly.io as pio
                    import json
                    
                    spec_json = viz.get("spec")
                    if isinstance(spec_json, str):
                        fig = pio.from_json(spec_json)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Could not display {viz_type}")
                
                else:
                    st.info(f"Visualization type: {viz_type}")
            
            except Exception as e:
                st.warning(f"Could not display visualization: {str(e)}")


def display_suggested_followups(followups: list):
    """Display suggested follow-up questions"""
    if not followups:
        return
    
    st.subheader("💡 Suggested Follow-ups")
    
    cols = st.columns(len(followups))
    for idx, followup in enumerate(followups[:3]):  # Limit to 3 suggestions
        with cols[idx]:
            if st.button(followup, key=f"followup_{idx}", use_container_width=True):
                st.session_state.qa_followup = followup
                st.rerun()


def display_conversation_context():
    """Display conversation context window"""
    memory = st.session_state.conversation_memory
    
    if not memory.messages:
        return
    
    with st.expander("📜 Conversation Memory", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Messages", len(memory.messages))
        with col2:
            st.metric("Topics", len(set([m.get("metadata", {}).get("query_type") for m in memory.messages])))
        with col3:
            avg_confidence = sum([m.get("metadata", {}).get("confidence", 0.5) for m in memory.messages]) / max(len(memory.messages), 1)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        st.markdown("---")
        st.write("**Recent Messages:**")
        
        # Display recent messages
        recent = memory.messages[-6:]  # Last 6 messages
        for msg in recent:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:100]  # Truncate
            confidence = msg.get("metadata", {}).get("confidence", 0)
            
            if role == "USER":
                st.write(f"👤 **User:** {content}...")
            else:
                st.write(f"🤖 **Assistant:** {content}... *(confidence: {confidence:.2f})*")


def render_qa_tab():
    """Main Q&A tab renderer"""
    
    try:
        # Initialize session
        initialize_qa_session()
        
        st.header("💬 Advanced Q&A with Conversation Memory")
        
        # Table selection section
        st.subheader("📁 Data Source Selection")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            processor = st.session_state.get("processor")
            available_tables = processor.list_tables() if processor else []
            
            # Create options with "All Tables" as default
            table_options = ["🌐 All Tables"] + available_tables
            
            if "qa_selected_table" not in st.session_state:
                st.session_state.qa_selected_table = "🌐 All Tables"
            
            selected_table = st.selectbox(
                "Select a CSV file to query (or search all):",
                table_options,
                index=table_options.index(st.session_state.qa_selected_table) if st.session_state.qa_selected_table in table_options else 0,
                key="qa_table_select"
            )
            st.session_state.qa_selected_table = selected_table
        
        with col2:
            st.write("")  # Spacing
            if selected_table != "🌐 All Tables":
                st.info(f"Searching: **{selected_table}** only")
            else:
                st.info(f"Searching: **All {len(available_tables)}** files")
        
        # Display system info
        with st.expander("ℹ️ System Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Conversation Messages", len(st.session_state.conversation_memory.messages))
            with col2:
                st.metric("Chat History", len(st.session_state.qa_history))
            with col3:
                st.metric("Memory Enabled", "✅" if st.session_state.conversation_memory else "❌")
        
        # Display conversation context
        display_conversation_context()
        
        st.divider()
        
        # Chat display area
        st.subheader("Chat History")
        
        chat_container = st.container(height=400, border=True)
        
        with chat_container:
            if not st.session_state.qa_history:
                st.info("No messages yet. Start by asking a question about the sales data!")
            else:
                for message in st.session_state.qa_history:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    confidence = message.get("confidence", 0)
                    table_source = message.get("table_source", "")
                    
                    if role == "user":
                        with st.chat_message("user"):
                            st.write(content)
                            if table_source and table_source != "🌐 All Tables":
                                st.caption(f"📁 Source: {table_source}")
                    else:
                        with st.chat_message("assistant"):
                            # Display response summary
                            if isinstance(content, dict):
                                st.write(content.get("summary", content))
                            else:
                                st.write(content)
                            if table_source:
                                st.caption(f"📁 Queried: {table_source}")
                            
                            # Display confidence if available
                            if confidence > 0:
                                display_confidence_indicator(confidence)
        
        # Input section
        st.divider()
        st.subheader("Ask a Question")
        
        # Check for followup question in session state
        if hasattr(st.session_state, 'qa_followup') and st.session_state.qa_followup:
            user_input = st.session_state.qa_followup
            st.session_state.qa_followup = None
        else:
            user_input = st.text_area(
                "Enter your question about the retail sales data:",
                placeholder="E.g., What were total sales in Q4? Which product had highest growth? Show me regional trends.",
                height=80,
                key="qa_input"
            )
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        send_button = col2.button("Send 📤", use_container_width=True, key="send_btn")
        clear_button = col3.button("Clear 🗑️", use_container_width=True, key="clear_btn")
        export_button = col4.button("Export 📥", use_container_width=True, key="export_btn")
        
        if send_button:
            if user_input.strip():
                # Create detailed progress container
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Get selected table constraint
                        selected_table = st.session_state.qa_selected_table
                        table_constraint = None if selected_table == "🌐 All Tables" else selected_table
                        
                        # Process query
                        orchestrator = st.session_state.qa_orchestrator
                        
                        # Add context to query if table is selected
                        if table_constraint:
                            enhanced_query = f"{user_input} [CONTEXT: Search in {table_constraint} table only]"
                        else:
                            enhanced_query = user_input
                        
                        # Step 1: Query Resolution
                        status_text.text("⏳ Step 1/4: Parsing query with AI...")
                        progress_bar.progress(15)
                        
                        # Step 2: Data Extraction
                        status_text.text("⏳ Step 2/4: Extracting data from database...")
                        progress_bar.progress(40)
                        
                        # Step 3: Validation
                        status_text.text("⏳ Step 3/4: Validating results...")
                        progress_bar.progress(65)
                        
                        result = orchestrator.process_query(enhanced_query)
                        
                        # Verify result is a dictionary
                        if not isinstance(result, dict):
                            raise ValueError(f"Invalid result type: {type(result)}")
                        
                        # Step 4: Formatting
                        status_text.text("⏳ Step 4/4: Formatting response...")
                        progress_bar.progress(90)
                        
                        # Check if query processing was successful
                        if result.get("error"):
                            progress_container.empty()
                            st.error(f"❌ Query Error: {result.get('error')}")
                            return
                        
                        # Add user message
                        st.session_state.qa_history.append({
                            "role": "user",
                            "content": user_input,
                            "confidence": 1.0,
                            "table_source": selected_table
                        })
                        
                        # Extract response data
                        formatted_response = result.get("formatted_response", {})
                        confidence = result.get("confidence_score", 0.5)
                        
                        # Validate formatted response
                        if not isinstance(formatted_response, dict):
                            formatted_response = {"summary": str(formatted_response)}
                        
                        # Add assistant message
                        st.session_state.qa_history.append({
                            "role": "assistant",
                            "content": formatted_response,
                            "confidence": confidence,
                            "full_result": result,
                            "table_source": selected_table
                        })
                        
                        # Complete
                        progress_bar.progress(100)
                        status_text.text("✅ Query processed successfully!")
                        import time
                        time.sleep(0.5)
                        progress_container.empty()
                        st.rerun()
                    
                    except Exception as e:
                        progress_container.empty()
                        error_msg = f"Error processing query: {str(e)}"
                        print(f"Q&A CRASH: {error_msg}")  # Log to console
                        import traceback
                        traceback.print_exc()  # Print full stack trace
                        st.error(f"❌ {error_msg}")
                        st.info("Please try rephrasing your question or select a different table.")
            else:
                st.warning("Please enter a question")
        
        if clear_button:
            from data_processor import get_processor
            st.session_state.qa_history = []
            st.session_state.conversation_memory = ConversationMemory()
            processor = st.session_state.get("processor") or get_processor()
            
            if USE_LANGGRAPH:
                st.session_state.qa_orchestrator = LangGraphQueryAgent(
                    processor=processor,
                    conversation_memory=st.session_state.conversation_memory
                )
            else:
                st.session_state.qa_orchestrator = MultiAgentOrchestrator(
                    processor=processor,
                    conversation_memory=st.session_state.conversation_memory
                )
            st.info("✅ Conversation cleared")
            st.rerun()
        
        if export_button:
            # Export conversation to JSON
            export_data = {
                "messages": st.session_state.qa_history,
                "conversation_stats": {
                    "total_messages": len(st.session_state.qa_history),
                    "memory_size": len(st.session_state.conversation_memory.messages)
                }
            }
            
            import json
            export_json = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="Download Conversation",
                data=export_json,
                file_name="qa_conversation.json",
                mime="application/json"
            )
        
        # Display last response details if available
        if st.session_state.qa_history and st.session_state.qa_history[-1].get("role") == "assistant":
            last_response = st.session_state.qa_history[-1]
            confidence = last_response.get("confidence", 0)
            full_result = last_response.get("full_result")
            
            st.divider()
            st.subheader("📊 Response Analysis")
            
            # Confidence indicator
            display_confidence_indicator(confidence)
            
            st.divider()
            
            # Data and visualizations
            if full_result:
                extracted_data = full_result.get("extracted_data", {})
                formatted_response = full_result.get("formatted_response", {})
                
                # Display data table
                data = extracted_data.get("data")
                if data is not None and not data.empty:
                    display_data_table(data)
                    st.divider()
                
                # Display visualizations
                visualizations = formatted_response.get("visualizations", [])
                if visualizations:
                    display_visualizations(visualizations)
                    st.divider()
                
                # Display suggested follow-ups
                followups = formatted_response.get("suggested_followups", [])
                if followups:
                    display_suggested_followups(followups)
    
    except Exception as e:
        logger.error(f"Q&A TAB FATAL ERROR: {str(e)}", exc_info=True)
        print(f"Q&A TAB CRASH: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"❌ Q&A system encountered an error: {str(e)}")
        st.info("Please refresh the page and try again.")


if __name__ == "__main__":
    render_qa_tab()
