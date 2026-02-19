"""
LLM Configuration and Initialization
Supports both Gemini and OpenAI models
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from .config import LLM_PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY
from typing import Optional

def get_llm(provider: Optional[str] = None, temperature: float = 0.1):
    """
    Get LLM instance based on provider
    
    Args:
        provider: "gemini" or "openai" (defaults to config)
        temperature: Model temperature for randomness
    
    Returns:
        LLM instance
    """
    provider = provider or LLM_PROVIDER
    
    if provider.lower() == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env file")
        
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GEMINI_API_KEY,
            temperature=temperature,
            max_output_tokens=2048
        )
    
    elif provider.lower() == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        
        return ChatOpenAI(
            model="gpt-4-turbo",  # Fixed: gpt-4.1 doesn't exist
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=2048
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# Get default LLM instance
def get_default_llm():
    """Get the default LLM configured in config.py"""
    return get_llm(temperature=0.1)
