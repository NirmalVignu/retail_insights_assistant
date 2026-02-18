# Retail Insights Assistant - Production Structure

## 📁 Project Structure

```
retail_insights_assistant/
├── app_new.py                      # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # Environment configuration (not for submission)
├── .env.example                    # API key template for submission
├── .gitignore                      # Git ignore rules
├── README.md                       # Main documentation
├── Retail_Insights_Assistant_Screenshots.pdf  # Screenshots & demo documentation
│
├── src/                            # Source code (organized)
│   ├── __init__.py
│   │
│   ├── agents/                     # Multi-agent system
│   │   ├── __init__.py
│   │   ├── multi_agent.py          # 4 specialized agents
│   │   └── summarization_engine.py # Business intelligence reports
│   │
│   ├── graph/                      # LangGraph workflow
│   │   ├── __init__.py
│   │   ├── langgraph_agent.py      # 7-node state machine
│   │   └── enhanced_query_resolution.py # SQL query architect
│   │
│   ├── utils/                      # Utilities
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration constants
│   │   ├── llm_config.py           # LLM provider setup
│   │   ├── data_processor.py       # DuckDB data engine
│   │   ├── conversation_memory.py  # RAG-based memory
│   │   ├── response_formatter.py   # Response formatting
│   │   ├── input_loader.py         # File loading utilities
│   │   ├── pdf_report_generator.py # PDF export
│   │   └── prompt_loader.py        # Prompt management
│   │
│   └── ui/                         # UI components
│       ├── __init__.py
│       └── qa_tab.py              # Q&A tab component
│
├── prompts/                        # Externalized prompt templates
│   ├── query_resolution_prompt.txt       # Query intent analysis
│   ├── query_decomposition_prompt.txt    # Complex query breakdown
│   ├── llm_analysis_prompt.txt           # LLM-based insights
│   ├── validation_analyst_prompt.txt     # Result validation
│   ├── data_analyst_prompt.txt           # Statistical analysis
│   ├── summarization_prompt.txt          # Business reports
│   ├── comparative_summarization_prompt.txt  # Multi-table comparison
│   └── comparison_prompt.txt             # Table comparison
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # System architecture (15 slides)  
│   ├── LANGGRAPH_VISUALIZATION.md # LangGraph workflow diagram
│   ├── AGENT_COMPARISON.md        # Dual agent architecture guide
│   ├── PROJECT_STRUCTURE.md       # This file
│   └── SCALABILITY_DESIGN.md      # 100GB+ scaling guide
│
├── screenshots/                   # Application screenshots (14 images)
│   ├── 01_landing_page.png.jpg
│   ├── 02_summary_page.jpg
│   ├── 03_QA_Chat_LG_*.jpg
│   ├── 04_data_analyst_*.jpg
│   ├── 05_data_explorer.jpg
│   └── langgraph_viz.jpg         # LangGraph workflow visualization
│
├── data/                          # Sample datasets
│   └── Sales Dataset/
│       ├── Amazon Sale Report.csv
│       ├── Sale Report.csv
│       ├── International sale Report.csv
│       └── May-2022.csv
│
├── config/                        # Configuration (optional)
│   └── (reserved for future config files)
│
└── tests/                         # Unit tests (future)
    └── (reserved for test files)
```

## 🚀 Quick Start

```bash
# Run the application
streamlit run app_new.py
```

## 📦 Installation

```bash
# 1. Clone/Download project
cd retail_insights_assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run application
streamlit run app_new.py
```

## 🎯 Key Features

### 1. **Organized Code Structure** 
- Agents in `src/agents/`
- LangGraph workflow in `src/graph/`
- Utilities in `src/utils/`
- UI components in `src/ui/`

### 2. **Externalized Prompts**
- All LLM prompts in `prompts/` folder
- Easy to edit without touching code
- Version control friendly
- Supports A/B testing and experimentation

### 3. **Professional Imports**
```python
# Organized modular imports
from src.agents import MultiAgentOrchestrator
from src.graph import LangGraphAgent
from src.utils import load_prompt, format_prompt
```

### 4. **Dynamic Prompt Management**
```python
# Load prompt from file
from src.utils import load_prompt, format_prompt

# Simple loading
prompt = load_prompt("summarization_prompt")

# With variable substitution
prompt = format_prompt(
    "query_resolution_prompt",
    schema_info=schema_data,
    context_part=previous_conversation
)
```

---

## 🖥️ Streamlit UI - Complete Tab Guide

The application has **4 interactive tabs**, each serving a distinct purpose in the analytics workflow.

### Tab 1: 📈 Automated Data Summarization

**Purpose**: Generate comprehensive business intelligence reports from loaded datasets.

**Features**:
- **Single Table Summary**:
  - Executive summary with key metrics
  - Quantified insights (actual numbers, no placeholders)
  - Top categories/products with revenue breakdown
  - Trend analysis with growth rates
  - Strategic recommendations based on data patterns
  
- **Comparative Analysis**:
  - Side-by-side comparison of two datasets
  - Dimensional overlap analysis
  - Integration opportunities identification
  - Data consolidation roadmap
  
- **PDF Export**: Download generated reports as professional PDF documents

**Typical Use Cases**:
- Generate monthly performance reports
- Compare sales across different time periods
- Analyze multiple data sources for integration planning

**Technology**: Uses `SummarizationEngine` with external prompts for consistent, high-quality reports.

---

### Tab 2: 💬 Conversational Q&A

**Purpose**: Ask natural language questions about your data and get instant, accurate answers.

**Dual Agent Architecture**:

#### 🧠 **LangGraph Agent** (Advanced Mode)
**Best for**: Complex, multi-step queries requiring reasoning

**Architecture**: 7-node state machine workflow
- Node 1: `analyze_query` - Parse user intent
- Node 2: `decompose_query` - Break complex questions into sub-queries
- Node 3: `extract_data` - Execute SQL and fetch results
- Node 4: `validate_results` - Quality checks and routing
- Node 5: `refine_query` - Optimize SQL if needed (max 2 retries)
- Node 6: `llm_analysis` - Generate insights from data
- Node 7: `format_response` - Structure final answer

**Use When**: 
- Comparing multiple time periods or categories
- Questions requiring multi-step calculations
- Trend analysis with pattern detection
- Queries needing error recovery

**Example Questions**:
- "Compare sales trends between Electronics and Home categories over Q1 and Q2"
- "What's the correlation between discount rates and sales volume by region?"
- "Show me products with declining sales but increasing returns"

#### 🤖 **Multi-Agent Orchestrator** (Fast Mode)
**Best for**: Simple, direct aggregation queries

**Architecture**: 4-agent linear pipeline
- Agent 1: `QueryResolutionAgent` - NLP to SQL mapping
- Agent 2: `DataExtractionAgent` - Query execution
- Agent 3: `ValidationAgent` - Result verification
- Agent 4: `DataAnalystAgent` - Statistical analysis

**Use When**:
- Quick metrics lookup (totals, averages, counts)
- Dashboard-style queries
- Simple filtering and grouping

**Example Questions**:
- "What is the total revenue by category?"
- "Which city has the highest order volume?"
- "Show me top 10 products by sales"
- "What's the average order value in May?"

#### **Common Features** (Both Agents):
- ✅ Confidence scoring (0.0-1.0) for every answer
- ✅ Conversation memory with FAISS + RAG
- ✅ Suggested follow-up questions
- ✅ Auto-generated visualizations (charts, tables)
- ✅ Real numeric values (no placeholder variables)
- ✅ Data source selection (query specific tables or all tables)
- ✅ Clear conversation history option

**Technology**: 
- LangGraph: `src/graph/langgraph_agent.py` with external prompts
- Multi-Agent: `src/agents/multi_agent.py` with 4 specialized agents
- Memory: FAISS vector store for semantic conversation search

---

### Tab 3: 📋 Data Explorer

**Purpose**: Quick, AI-free exploration and inspection of loaded datasets.

**Features**:
1. **Table Selection**: Dropdown to choose any loaded table
2. **Metadata Display**:
   - Total row count
   - Total column count
   - Table name and file source
3. **Data Preview**:
   - First 100 rows displayed in interactive table
   - Sortable columns
   - Scrollable view
4. **Column Profiling**:
   - Column names list
   - Data types for each column
   - Missing value indicators
5. **Basic Statistics**:
   - Numeric column ranges (min/max)
   - Record counts
   - Data completeness overview

**Typical Use Cases**:
- Verify data loaded correctly
- Quick sanity checks before analysis
- Understand data structure and column names
- Spot obvious data quality issues

**Technology**: Direct DuckDB queries via `DataProcessor`, no LLM calls (fast and free).

**Sub-tabs**:
- **Data View**: Raw data table display
- **Visualizations**: Quick charts (distributions, time series)
- **Analytics**: Basic aggregations and groupings

---

### Tab 4: 🔬 Data Analyst

**Purpose**: Professional, comprehensive statistical analysis and data quality assessment.

**Features**:
1. **Executive Overview**:
   - Business domain identification
   - Data maturity assessment
   - Dataset complexity scoring
   - Recommended analysis depth

2. **Statistical Findings**:
   - Distribution analysis for all numeric columns
   - Range analysis (min, max, median, quartiles)
   - Variability metrics (standard deviation, coefficient of variation)
   - Correlation detection between numeric fields

3. **Data Quality Assessment**:
   - Completeness score (% of non-null values)
   - Duplicate detection
   - Consistency checks
   - Missing value patterns

4. **Anomaly Detection**:
   - **Critical**: Outliers requiring immediate attention (>3 std dev)
   - **Moderate**: Notable deviations (2-3 std dev)
   - **Minor**: Slight variations (1-2 std dev)
   - Impact assessment for each anomaly

5. **Categorical Insights**:
   - Top categories by frequency
   - Category distribution (concentration vs diversity)
   - Rare category identification
   - Category completeness

6. **Actionable Recommendations**:
   - Prioritized action items (High/Medium/Low)
   - Data cleaning suggestions
   - Analysis opportunities
   - Integration considerations

**Interactive Visualizations** (4 sub-tabs):
- **Distributions**: Histograms for all numeric columns
- **Categories**: Bar charts for categorical breakdowns
- **Correlation Matrix**: Heatmap showing variable relationships
- **Box Plots**: Outlier visualization and quartile ranges

**Typical Use Cases**:
- Pre-analysis data profiling
- Data quality audits
- Identifying data preparation needs
- Spotting unusual patterns or errors

**Technology**: 
- Statistical analysis via pandas + DuckDB aggregations
- Visualization with Plotly interactive charts
- Uses `DataAnalystAgent` in `src/agents/multi_agent.py`
- Driven by `data_analyst_prompt.txt` for consistent professional reporting

---

## 📝 Usage Examples

### Import from New Structure
```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import agents
from src.agents import MultiAgentOrchestrator, DataAnalystAgent
from src.graph import LangGraphAgent
from src.utils import get_processor, get_llm, load_prompt

# Use in code
agent = LangGraphAgent(processor=get_processor())
result = agent.process_query("What is total revenue?")
```

### Edit Prompts
1. Open `prompts/query_resolution_prompt.txt`
2. Edit the text directly
3. Save file
4. Changes take effect immediately (cached prompts can be reloaded)

### Add New Prompt
1. Create `prompts/my_new_prompt.txt`
2. Write prompt content
3. Load in code: prompt_loader.load_prompt("my_new_prompt")

## 🔧 Configuration

### Environment Variables (.env)
```bash
# LLM Provider (gemini or openai)
LLM_PROVIDER=gemini

# API Keys
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Database
DATABASE_PATH=sales_data.duckdb
CSV_DATA_PATH=Assignment/Sales Dataset/

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Config File (src/utils/config.py)
All configuration constants in one place:
- `PAGE_TITLE`
- `DATABASE_PATH`
- `CSV_DATA_PATH`
- `EMBEDDING_MODEL`

## 🧪 Testing

```bash
# Test imports
python -c "from src.agents import MultiAgentOrchestrator; print('✓ Imports work!')"

# Test prompt loading
python -c "from src.utils import load_prompt; print(load_prompt('summarization_prompt')[:100])"

# Run application
streamlit run app_new.py
```

## 📚 Documentation Files

- **Setup Guide**: `README.md` - Installation, configuration, usage instructions
- **Architecture**: `docs/ARCHITECTURE.md` - 15-slide technical architecture + LangGraph visualization
- **Agent Comparison**: `docs/AGENT_COMPARISON.md` - Dual agent system explained
- **LangGraph Workflow**: `docs/LANGGRAPH_VISUALIZATION.md` - 7-node state machine diagram
- **Scalability**: `docs/SCALABILITY_DESIGN.md` - 100GB+ production design
- **Screenshots**: `Retail_Insights_Assistant_Screenshots.pdf` - Complete demo with 14 images
- **Project Structure**: `docs/PROJECT_STRUCTURE.md` - This file



## 🐛 Troubleshooting

### Import Errors
```python
# Make sure src/ is in path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

### Prompt Not Found
```python
# Check prompts/ directory exists
# Check filename matches (case-sensitive)
# Check .txt extension

from src.utils import get_prompt_loader
loader = get_prompt_loader()
print(loader.list_available_prompts())  # List all prompts
```

### Module Not Found
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify Python version (3.9+)
python --version
```

### Application Won't Start
```bash
# Clear cache
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue __pycache__, src\__pycache__, src\*\__pycache__

# Restart Streamlit
streamlit run app_new.py
```

## 📞 Support

For issues:
1. Check main `README.md` troubleshooting section
2. Review error logs in terminal
3. Verify `.env` configuration is correct

