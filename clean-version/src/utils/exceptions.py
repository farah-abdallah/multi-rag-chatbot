"""
Custom exceptions for the Multi-RAG Chatbot application.
"""


class MultiRAGError(Exception):
    """Base exception for Multi-RAG Chatbot."""
    pass


class CRAGError(MultiRAGError):
    """Exception raised during CRAG processing."""
    pass


class DocumentProcessingError(MultiRAGError):
    """Exception raised during document processing."""
    pass


class LLMError(MultiRAGError):
    """Exception raised during LLM operations."""
    pass


class SearchError(MultiRAGError):
    """Exception raised during search operations."""
    pass


class RetrievalError(MultiRAGError):
    """Exception raised during information retrieval."""
    pass


class EvaluationError(MultiRAGError):
    """Exception raised during evaluation processes."""
    pass


class KnowledgeError(MultiRAGError):
    """Exception raised during knowledge processing."""
    pass


class APIError(MultiRAGError):
    """Exception raised during API operations."""
    pass


class ConfigurationError(MultiRAGError):
    """Exception raised due to configuration issues."""
    pass


class ValidationError(MultiRAGError):
    """Exception raised during data validation."""
    pass
