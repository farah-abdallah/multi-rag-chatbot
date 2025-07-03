"""
Configuration settings for the Multi-RAG Chatbot application.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # API Configuration - Use single key approach to avoid parsing issues
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    
    # Rate limiting configuration
    rate_limit_per_key: int = Field(default=100, env="RATE_LIMIT_PER_KEY")
    cooldown_after_limit: int = Field(default=60, env="COOLDOWN_AFTER_LIMIT")
    min_request_interval: float = Field(default=0.6, env="MIN_REQUEST_INTERVAL")
    
    # CRAG Settings
    crag_web_search: bool = Field(default=True, env="CRAG_WEB_SEARCH")
    crag_fallback_mode: bool = Field(default=True, env="CRAG_FALLBACK_MODE")
    
    # LLM Configuration
    default_model: str = Field(default="gemini-1.5-flash", env="DEFAULT_MODEL")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    max_tokens: int = Field(default=2000, env="MAX_TOKENS")
    top_p: float = Field(default=0.95, env="TOP_P")
    top_k: int = Field(default=40, env="TOP_K")
    
    # Document Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_documents: int = Field(default=50, env="MAX_DOCUMENTS")
    
    # Search Configuration
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS")
    search_timeout: int = Field(default=30, env="SEARCH_TIMEOUT")
    
    # UI Configuration
    page_title: str = Field(default="Multi-RAG Chatbot", env="PAGE_TITLE")
    page_icon: str = Field(default="🤖", env="PAGE_ICON")
    sidebar_width: int = Field(default=300, env="SIDEBAR_WIDTH")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    
    # Development
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # File paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables
    
    @property
    def valid_api_keys(self) -> List[str]:
        """Return valid API keys from environment variables."""
        # First try to get from GOOGLE_API_KEYS (comma-separated)
        api_keys_str = os.getenv("GOOGLE_API_KEYS", "")
        if api_keys_str:
            keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
            if keys:
                return keys
        
        # Fall back to single key
        single_key = self.google_api_key or os.getenv("GOOGLE_API_KEY", "")
        if single_key:
            return [single_key]
        
        return []
    
    def model_post_init(self, __context) -> None:
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()
