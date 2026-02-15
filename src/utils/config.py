import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # Options: "gemini", "openai"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Database Configuration
DATABASE_PATH = "sales_data.db"
DUCKDB_PATH = "sales_data.duckdb"

# Data Configuration
CSV_DATA_PATH = "data/Sales Dataset"
SUPPORTED_CSV_FILES = [
    "Amazon Sale Report.csv",
    "Sale Report.csv",
    "International sale Report.csv",
    "May-2022.csv",
]

# Agent Configuration
AGENT_TIMEOUT = 30
MAX_ITERATIONS = 10

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Streamlit Configuration
PAGE_TITLE = "Retail Insights Assistant"
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"
