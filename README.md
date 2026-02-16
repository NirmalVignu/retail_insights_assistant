w# Retail Insights Assistant 🛍️📊

**GenAI-Powered Multi-Agent System for Retail Analytics**

An intelligent retail analytics assistant that combines LangGraph-based multi-agent architecture with advanced LLM capabilities to provide automated data summarization, conversational Q&A, and comprehensive dataset analysis.

---

## 🎯 Features

### 1. **Automated Data Summarization** 📈
- Generate comprehensive business intelligence reports from sales data
- Quantified insights with actual metrics and KPIs
- Comparative analysis between multiple datasets
- Export summaries as PDF reports

### 2. **Conversational Q&A** 💬

**Two Agent Options Available:**

#### 🧠 **LangGraph Agent (Advanced)**
- **Architecture**: 7-node state machine workflow
- **Best For**: Complex queries requiring multi-step reasoning
- **Features**:
  - Automatic query decomposition for complex questions
  - Intelligent query refinement with validation loops
  - Sub-query execution and result merging
  - Advanced error recovery with retry mechanisms
  - Detailed execution tracing
- **Use When**: You need to ask complex questions like "Compare sales trends across categories and identify outliers"

#### 🤖 **Multi-Agent Orchestrator (Fast)**
- **Architecture**: 4-agent linear pipeline
- **Best For**: Simple, direct questions requiring quick answers
- **Features**:
  - Streamlined 4-stage processing (Resolution → Extraction → Validation → Formatting)
  - Faster response times
  - Direct computation-based answers for aggregation queries
  - Built-in answer patterns for common questions
- **Use When**: You need quick answers like "What's the total revenue?" or "Which category has highest sales?"

**Common Features (Both Agents):**
- Natural language query understanding
- Conversation memory with RAG (Retrieval-Augmented Generation)
- Confidence scoring for all responses
- Data source selection (query specific tables or all tables)
- Suggested follow-up questions
- Actual numeric values (no placeholder variables)

### 3. **Data Explorer** 📋
- Interactive data preview for all loaded tables
- Table statistics and metadata
- Column profiling and data type analysis
- Quick data quality assessment

### 4. **Data Analyst** 🔬
- Professional dataset profiling
- Statistical analysis with outlier detection
- Data quality assessment with completeness scoring
- Anomaly identification (Critical/Moderate/Minor)
- Actionable recommendations
- Interactive visualizations (distributions, correlations, box plots)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app.py)                    │
│  Tab1: Summarization │ Tab2: Q&A │ Tab3: Explorer │ Tab4: Analyst │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           │                                     │
    ┌──────▼────────┐                  ┌────────▼─────────┐
    │  LangGraph    │                  │  Multi-Agent     │
    │  Q&A Agent    │                  │  Orchestrator    │
    └───────┬───────┘                  └────────┬─────────┘
            │                                   │
    ┌───────▼────────────────────────────────┬─┴──────────┐
    │  Enhanced Query Resolution             │            │
    │  (Semantic Understanding + SQL)        │            │
    └───────┬────────────────────────────────┘            │
            │                                              │
    ┌───────▼──────────┐                     ┌────────────▼──────┐
    │   Data Processor │◄────────────────────┤ Summarization     │
    │   (DuckDB)       │                     │ Engine            │
    └───────┬──────────┘                     └────────────┬──────┘
            │                                              │
    ┌───────▼──────────────────────────────────────────────▼──────┐
    │              LLM Layer (Gemini / OpenAI)                    │
    │           + Conversation Memory (FAISS + RAG)              │
    └────────────────────────────────────────────────────────────┘
```

### Multi-Agent LangGraph Workflow
```
User Query
    │
    ▼
┌─────────────────┐
│ Analyze Query   │ ◄──┐
└────────┬────────┘    │
         │             │
    ▼ Complex?         │
         │             │
┌────────▼────────┐    │
│ Decompose Query │    │ Refine
└────────┬────────┘    │ Query
         │             │
┌────────▼────────┐    │
│ Extract Data    │    │
└────────┬────────┘    │
         │             │
┌────────▼────────┐    │
│ LLM Analysis    │    │
└────────┬────────┘    │
         │             │
┌────────▼────────┐    │
│ Validate Results│────┘
└────────┬────────┘
         │
    ▼ Valid?
         │
┌────────▼────────┐
│ Format Response │
└────────┬────────┘
         │
         ▼
    User Response
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- API keys for Gemini or OpenAI (at least one required)

### Step 1: Clone/Extract Project
```bash
cd d:/Blend_360/retail_insights_assistant
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `langchain>=0.1.0` - LLM framework
- `langchain-google-genai>=0.0.8` - Gemini integration
- `langchain-openai>=0.0.5` - OpenAI integration
- `langgraph>=0.0.20` - Multi-agent orchestration
- `streamlit>=1.28.0` - Web UI
- `duckdb>=0.8.0` - In-memory analytical database
- `pandas>=2.0.0` - Data manipulation
- `plotly>=5.17.0` - Interactive visualizations
- `faiss-cpu>=1.7.4` - Vector similarity search
- `sentence-transformers>=2.2.0` - Text embeddings
- `reportlab>=4.0.0` - PDF generation

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Provider Configuration
LLM_PROVIDER=gemini          # Options: "gemini" or "openai"

# API Keys (provide at least one)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**How to get API keys:**

- **Gemini API**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- **OpenAI API**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)

### Step 5: Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📁 Project Structure

```
retail_insights_assistant/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── .gitignore                     # Git ignore patterns
│
├── Core Modules:
├── langgraph_agent.py             # LangGraph Q&A agent with tools
├── multi_agent.py                 # Multi-agent orchestrator (4 agents)
├── enhanced_query_resolution.py   # NLP to SQL query resolution
├── data_processor.py              # DuckDB data operations
├── summarization_engine.py        # Business intelligence reports
├── response_formatter.py          # Response formatting & visualizations
├── conversation_memory.py         # RAG-based conversation memory
│
├── Utilities:
├── input_loader.py                # CSV/Excel/JSON data loading
├── llm_config.py                  # LLM provider configuration
├── pdf_report_generator.py        # PDF export functionality
├── qa_tab.py                      # Q&A tab helpers
│
└── Documentation:
    ├── README.md                   # This file
    ├── ARCHITECTURE.md             # System architecture & scalability
    ├── SCALABILITY_DESIGN.md       # 100GB+ scaling design
    └── SCREENSHOTS_GUIDE.md        # UI documentation guide
```

---

## 🎮 Usage Guide

### 1. Upload Data

**Supported formats:**
- CSV files (recommended)
- Excel files (.xlsx)
- JSON files
- Multiple file upload supported

**From the sidebar:**
1. Click "Browse files" button
2. Select one or more data files
3. Click "Load Data" button
4. Wait for confirmation message

**Example data structure:**
```csv
Order ID,Date,Category,Amount,Qty,Status,ship-city
101,2022-01-15,Electronics,1250.00,2,Shipped,New York
102,2022-01-16,Clothing,450.75,3,Pending,Los Angeles
```

### 2. Tab 1: Automated Summarization

**Generate comprehensive business reports:**

1. Select a loaded table from dropdown
2. Click "Generate Summary" button
3. Review the generated report with:
   - Executive Summary
   - Key Metrics (with actual values)
   - Key Insights (data-driven patterns)
   - Strategic Recommendations
4. Export as PDF using "Download PDF Report" button

**Comparison Mode:**
1. Toggle "Compare Two Tables"
2. Select two tables to compare
3. Get dimensional comparison, integration opportunities, and roadmap

### 3. Tab 2: Q&A Chat (Multi-Agent System)

**Ask questions in natural language:**

**Example questions:**
- "What is the total revenue by category?"
- "Which region had the highest sales growth?"
- "Show me monthly trends for the last quarter"
- "What is the average order value by customer segment?"
- "Compare Q1 vs Q2 performance"

**Features:**
- Confidence scores for each response
- Suggested follow-up questions
- Data visualizations (auto-generated)
- Conversation history with memory
- Clear conversation option

**Chat Controls:**
- Type question in input box
- Press Enter or click "Send"
- View response with data tables and charts
- Click suggested questions to ask quickly
- Use "Clear Memory" to reset conversation

### 4. Tab 3: Data Explorer

**Inspect loaded data:**

1. Select table from dropdown
2. View:
   - Row and column counts
   - Data preview (first 100 rows)
   - Column names and types
   - Basic statistics

### 5. Tab 4: Data Analyst

**Professional dataset profiling:**

1. Select table from dropdown
2. Click "Analyze Dataset"
3. Review comprehensive analysis:
   - **Executive Overview**: Business domain, data maturity
   - **Statistical Findings**: Distributions, ranges, variability
   - **Data Quality Assessment**: Completeness score, duplicates
   - **Anomalies**: Outlier detection (Critical/Moderate/Minor)
   - **Categorical Insights**: Top categories, diversity metrics
   - **Recommendations**: Prioritized action items
4. View automated visualizations:
   - Numeric distributions
   - Category value counts
   - Correlation heatmap
   - Box plots for outliers
   - Missing data analysis

---

## � Agent Comparison: LangGraph vs Multi-Agent

### When to Use Each Agent

| Feature | LangGraph (Advanced) | Multi-Agent (Fast) |
|---------|---------------------|-------------------|
| **Processing Time** | 3-8 seconds | 1-3 seconds |
| **Complex Queries** | ✅ Excellent | ⚠️ Limited |
| **Simple Queries** | ✅ Good | ✅ Excellent |
| **Query Decomposition** | ✅ Automatic | ❌ No |
| **Error Recovery** | ✅ Self-healing with retries | ⚠️ Basic |
| **Validation Loops** | ✅ Multi-stage | ✅ Single-stage |
| **Best For** | Research, exploration | Quick facts, dashboards |

### Architectural Differences

**LangGraph Agent (7 Nodes):**
```
Query → Analyze → [Decompose?] → Extract → LLM Analysis → Validate → [Refine?] → Format
```
- State machine with conditional branching
- Can loop back to refine queries
- Handles complex multi-table joins
- Breaks down compound questions automatically

**Multi-Agent Orchestrator (Linear Pipeline):**
```
Query → Resolution → Extraction → Validation → Response
```
- Straightforward 4-stage pipeline
- Direct computation for aggregations (SUM, COUNT, etc.)
- Pattern matching for common questions
- Optimized for single-table queries

### Example Use Cases

**Use LangGraph For:**
- "Compare sales trends between Electronics and Home categories over the last 3 months, and identify any unusual patterns"
- "What's the correlation between discount rates and sales volume, segmented by region?"
- "Show me products with declining sales but increasing returns"

**Use Multi-Agent For:**
- "Which category has the highest total sales?"
- "What's the average order value?"
- "Show me top 10 products by revenue"
- "How many orders were placed in May?"

**💡 Tip**: Start with Multi-Agent for quick answers. Switch to LangGraph when queries get more complex or if the first attempt doesn't capture your full intent.

---

## �🔧 Configuration Options

### Change LLM Provider

Edit `.env` file:
```bash
# Use Gemini (default)
LLM_PROVIDER=gemini

# OR use OpenAI
LLM_PROVIDER=openai
```

### Adjust Model Temperature

Edit files directly for custom temperatures:
- `llm_config.py` - Default: 0.1 (deterministic)
- Higher values (0.7-0.9) = more creative responses
- Lower values (0.0-0.3) = more focused, deterministic

### Modify Agent Behavior

**Query Resolution** (`enhanced_query_resolution.py`):
- Edit column mapping rules (line 151-160)
- Adjust confidence thresholds
- Add domain-specific terminology

**Summarization** (`summarization_engine.py`):
- Customize report structure (line 80-125)
- Modify metric calculations
- Add custom KPIs

**LangGraph Workflow** (`langgraph_agent.py`):
- Adjust node processing logic
- Modify validation criteria
- Add custom tools

---

## 🧪 Testing & Validation

### Test with Sample Data

Sample datasets included in `data/Sales Dataset/`:
- `Amazon Sale Report.csv`
- `Sale Report.csv`
- `International sale Report.csv`
- `May-2022.csv`

### Example Test Queries

**Basic Analytics:**
```
"What is the total amount?"
"How many orders do we have?"
"Show me top 5 categories by revenue"
```

**Time-Series:**
```
"What is the monthly trend of orders?"
"Show revenue by month"
"Compare January vs February sales"
```

**Aggregations:**
```
"What is average order value by city?"
"Total quantity sold per category"
"Revenue distribution by status"
```

**Complex Queries:**
```
"Which category generates highest revenue, and what is the breakdown by region?"
"Show monthly growth rate of orders and revenue"
"Identify top performing products in Q1"
```

---

## 📊 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Gemini Pro / GPT-3.5 | Natural language understanding |
| **Orchestration** | LangGraph | Multi-agent workflow management |
| **Framework** | LangChain | LLM application framework |
| **Database** | DuckDB | In-memory analytical queries |
| **Vector Store** | FAISS | Semantic similarity search |
| **Embeddings** | Sentence Transformers | Text to vector conversion |
| **UI** | Streamlit | Interactive web interface |
| **Visualization** | Plotly | Interactive charts |
| **Data Processing** | Pandas | Data manipulation |

---

## ⚠️ Assumptions & Limitations

### Assumptions
1. **Data Format**: Structured tabular data (CSV/Excel/JSON)
2. **Schema Consistency**: Column names are relatively consistent
3. **Data Size**: Current implementation optimized for <10GB datasets
4. **Internet**: Active internet connection required for LLM API calls
5. **API Keys**: Valid API keys with sufficient quota

### Current Limitations
1. **Scale**: In-memory processing limits dataset size (~10GB)
2. **Real-time**: Not optimized for streaming data
3. **Multi-language**: Primarily English language queries
4. **Complex Joins**: Limited support for multi-table joins
5. **Historical Context**: Conversation memory limited to session

### Known Issues
- Large datasets (>5GB) may cause performance degradation
- Complex nested queries require manual decomposition
- PDF export may fail with extremely large tables
- FAISS indexing can be slow for first-time conversations

---

## 🚀 Future Improvements

### Short-term (Next Iteration)
- [ ] Add caching layer for frequent queries
- [ ] Implement query result pagination
- [ ] Add user authentication and session management
- [ ] Support for custom metric definitions
- [ ] Export data tables to Excel/CSV
- [ ] Dark mode UI theme

### Medium-term (Scalability Focus)
- [ ] Implement distributed processing (PySpark/Dask)
- [ ] Cloud data warehouse integration (BigQuery/Snowflake)
- [ ] Implement data lake architecture (S3/Azure Data Lake)
- [ ] Add streaming data support (Kafka/Kinesis)
- [ ] Build materialized views for common queries
- [ ] Implement query result caching (Redis)

### Long-term (Enterprise Features)
- [ ] Multi-tenant architecture
- [ ] Role-based access control (RBAC)
- [ ] Audit logging and compliance
- [ ] Real-time dashboards
- [ ] Scheduled report generation
- [ ] API endpoints for programmatic access
- [ ] Model fine-tuning on domain data
- [ ] Advanced anomaly detection with ML
- [ ] Predictive analytics capabilities

See **SCALABILITY_DESIGN.md** for detailed 100GB+ architecture.

---

## 🐛 Troubleshooting

### Issue: "API Key not found"
**Solution:** Ensure `.env` file exists with valid API keys
```bash
# Check .env file
cat .env

# Verify format
GEMINI_API_KEY=AIza...
```

### Issue: "Module not found"
**Solution:** Reinstall dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: "Port already in use"
**Solution:** Run on different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: "Out of memory"
**Solution:** Process smaller datasets or increase system RAM
- Reduce data size: `df.sample(frac=0.5)` for 50% sample
- Use DuckDB's LIMIT clause in queries
- Close other memory-intensive applications

### Issue: "Slow query responses"
**Solution:**
1. Check API rate limits (Gemini/OpenAI)
2. Reduce conversation memory size in `conversation_memory.py`
3. Simplify complex queries
4. Add indexes to DuckDB tables

### Issue: "Invalid SQL generation"
**Solution:**
1. Review query resolution prompts
2. Add column name mappings to `enhanced_query_resolution.py`
3. Use explicit column names in questions
4. Check data types match query operations

---


### Resources
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Project Documentation
- [Agent Comparison Guide](docs/AGENT_COMPARISON.md) - Detailed guide on LangGraph vs Multi-Agent
- [Architecture Presentation](docs/ARCHITECTURE.md) - 15-slide technical architecture
- [LangGraph Workflow Visualization](docs/LANGGRAPH_VISUALIZATION.md) - 7-node state machine diagram
- [Project Structure](docs/PROJECT_STRUCTURE.md) - Code organization guide
- [Scalability Design](docs/SCALABILITY_DESIGN.md) - 100GB+ scale architecture
- [Screenshots Guide](SCREENSHOTS_GUIDE.md) - UI feature documentation
- [DuckDB Documentation](https://duckdb.org/docs/)



---


**Built with ❤️ using LangGraph Multi-Agent Architecture**
