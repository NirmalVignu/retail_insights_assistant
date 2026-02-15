"""
Utility modules for configuration, data processing, and formatting
"""

from .config import (
    LLM_PROVIDER,
    DATABASE_PATH,
    CSV_DATA_PATH,
    EMBEDDING_MODEL
)
from .llm_config import get_llm
from .data_processor import DataProcessor, get_processor

# Optional import - requires faiss
try:
    from .conversation_memory import ConversationMemory
    _has_conversation_memory = True
except ImportError:
    ConversationMemory = None
    _has_conversation_memory = False

from .response_formatter import ResponseFormatter
from .input_loader import infer_table_name, load_dataframe_from_bytes, parse_text_summary

# Optional import - requires reportlab
try:
    from .pdf_report_generator import generate_pdf_report
    _has_pdf_generator = True
except ImportError:
    generate_pdf_report = None
    _has_pdf_generator = False

from .prompt_loader import PromptLoader, get_prompt_loader, load_prompt, format_prompt

__all__ = [
    # Config
    "LLM_PROVIDER",
    "DATABASE_PATH",
    "CSV_DATA_PATH",
    "EMBEDDING_MODEL",
    # LLM
    "get_llm",
    # Data Processing
    "DataProcessor",
    "get_processor",
    # Memory & Context
    "ConversationMemory",
    # Formatters
    "ResponseFormatter",
    "infer_table_name",
    "load_dataframe_from_bytes",
    "parse_text_summary",
    "generate_pdf_report",
    # Prompts
    "PromptLoader",
    "get_prompt_loader",
    "load_prompt",
    "format_prompt"
]
