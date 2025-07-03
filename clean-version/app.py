"""
Main entry point for the RAG Chatbot application.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.ui.streamlit_app import MultiRAGChatbot
from src.utils.logging import setup_logging
from config.settings import Settings


def main():
    """Main entry point for the application."""
    try:
        # Setup logging
        setup_logging()
        
        # Load configuration
        settings = Settings()
        
        # Create and run the app
        app = MultiRAGChatbot()
        app.run()
        
    except Exception as e:
        st.error(f"An error occurred while starting the application: {str(e)}")
        st.error("Please check the logs for more details.")


if __name__ == "__main__":
    main()
