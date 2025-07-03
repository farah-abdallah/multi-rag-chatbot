"""UI module for Multi-RAG Chatbot."""

from .streamlit_app import MultiRAGChatbot, main
from .components import *
from .styles import apply_custom_styles, get_theme_colors

__all__ = ['MultiRAGChatbot', 'main', 'apply_custom_styles', 'get_theme_colors']
