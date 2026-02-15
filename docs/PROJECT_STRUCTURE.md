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

