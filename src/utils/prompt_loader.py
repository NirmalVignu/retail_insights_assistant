"""
Prompt Loader Utility
Loads prompts from text files in the prompts/ directory
"""

import os
from pathlib import Path
from typing import Dict

class PromptLoader:
    """Load and manage prompts from external files"""
    
    def __init__(self, prompts_dir: str = None):
        """
        Initialize prompt loader
        
        Args:
            prompts_dir: Path to prompts directory (defaults to project_root/prompts/)
        """
        if prompts_dir is None:
            # Get project root (parent of src/)
            current_file = Path(__file__).resolve()
            # Go up from utils/ -> src/ -> project_root/
            project_root = current_file.parent.parent.parent
            prompts_dir = project_root / "prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt from file
        
        Args:
            prompt_name: Name of prompt file (without .txt extension)
        
        Returns:
            Prompt content as string
        """
        # Check cache first
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        # Load from file
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
        
        # Cache for future use
        self._cache[prompt_name] = prompt_content
        
        return prompt_content
    
    def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Load and format a prompt with variables
        
        Args:
            prompt_name: Name of prompt file
            **kwargs: Variables to format into the prompt
        
        Returns:
            Formatted prompt string
        """
        prompt_template = self.load_prompt(prompt_name)
        
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {e}")
    
    def list_available_prompts(self) -> list:
        """List all available prompt files"""
        if not self.prompts_dir.exists():
            return []
        
        return [f.stem for f in self.prompts_dir.glob("*.txt")]
    
    def reload_prompt(self, prompt_name: str) -> str:
        """Force reload a prompt from disk (bypass cache)"""
        if prompt_name in self._cache:
            del self._cache[prompt_name]
        return self.load_prompt(prompt_name)
    
    def clear_cache(self):
        """Clear all cached prompts"""
        self._cache.clear()


# Global instance for easy import
_prompt_loader = None

def get_prompt_loader() -> PromptLoader:
    """Get global prompt loader instance"""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader


# Convenience functions
def load_prompt(prompt_name: str) -> str:
    """Load a prompt by name"""
    return get_prompt_loader().load_prompt(prompt_name)


def format_prompt(prompt_name: str, **kwargs) -> str:
    """Load and format a prompt with variables"""
    return get_prompt_loader().format_prompt(prompt_name, **kwargs)
