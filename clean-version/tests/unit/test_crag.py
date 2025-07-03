"""
Unit tests for CRAG (Corrective Retrieval Augmented Generation) functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.crag import CRAGProcessor
from src.utils.exceptions import RetrievalError, LLMError


class TestCRAGProcessor:
    """Test cases for CRAGProcessor class."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_evaluator = Mock()
        mock_web_search = Mock()
        return mock_retriever, mock_llm, mock_evaluator, mock_web_search
    
    @pytest.fixture
    def crag_processor(self, mock_components):
        """Create CRAGProcessor instance with mock components."""
        retriever, llm, evaluator, web_search = mock_components
        return CRAGProcessor(
            retriever=retriever,
            llm=llm,
            evaluator=evaluator,
            web_search=web_search
        )
    
    def test_init(self, crag_processor):
        """Test CRAGProcessor initialization."""
        assert crag_processor.retriever is not None
        assert crag_processor.llm is not None
        assert crag_processor.evaluator is not None
        assert crag_processor.web_search is not None
    
    def test_process_query_with_good_retrieval(self, crag_processor, mock_components):
        """Test query processing with good retrieval results."""
        retriever, llm, evaluator, web_search = mock_components
        
        # Mock retrieval
        mock_docs = [
            {"content": "Test document 1", "metadata": {"source": "doc1.pdf"}},
            {"content": "Test document 2", "metadata": {"source": "doc2.pdf"}}
        ]
        retriever.retrieve.return_value = mock_docs
        
        # Mock evaluation (good quality)
        evaluator.evaluate_retrieval.return_value = {
            "quality": "good",
            "relevance_score": 0.8,
            "confidence": 0.9
        }
        
        # Mock LLM response
        llm.generate.return_value = "Generated response based on documents"
        
        query = "Test query"
        result = crag_processor.process_query(query)
        
        assert result["response"] == "Generated response based on documents"
        assert result["sources"] == mock_docs
        assert result["evaluation"]["quality"] == "good"
        retriever.retrieve.assert_called_once_with(query)
        evaluator.evaluate_retrieval.assert_called_once()
        llm.generate.assert_called_once()
        web_search.search.assert_not_called()
    
    def test_process_query_with_poor_retrieval(self, crag_processor, mock_components):
        """Test query processing with poor retrieval results that trigger web search."""
        retriever, llm, evaluator, web_search = mock_components
        
        # Mock retrieval
        mock_docs = [{"content": "Irrelevant document", "metadata": {"source": "doc1.pdf"}}]
        retriever.retrieve.return_value = mock_docs
        
        # Mock evaluation (poor quality)
        evaluator.evaluate_retrieval.return_value = {
            "quality": "poor",
            "relevance_score": 0.2,
            "confidence": 0.3
        }
        
        # Mock web search
        web_docs = [{"content": "Web search result", "metadata": {"source": "web"}}]
        web_search.search.return_value = web_docs
        
        # Mock LLM response
        llm.generate.return_value = "Response based on web search"
        
        query = "Test query"
        result = crag_processor.process_query(query)
        
        assert result["response"] == "Response based on web search"
        assert result["sources"] == web_docs
        assert result["evaluation"]["quality"] == "poor"
        retriever.retrieve.assert_called_once_with(query)
        evaluator.evaluate_retrieval.assert_called_once()
        web_search.search.assert_called_once_with(query)
        llm.generate.assert_called_once()
    
    def test_process_query_with_mixed_retrieval(self, crag_processor, mock_components):
        """Test query processing with mixed retrieval results."""
        retriever, llm, evaluator, web_search = mock_components
        
        # Mock retrieval
        mock_docs = [
            {"content": "Good document", "metadata": {"source": "doc1.pdf"}},
            {"content": "Poor document", "metadata": {"source": "doc2.pdf"}}
        ]
        retriever.retrieve.return_value = mock_docs
        
        # Mock evaluation (mixed quality)
        evaluator.evaluate_retrieval.return_value = {
            "quality": "mixed",
            "relevance_score": 0.5,
            "confidence": 0.6,
            "filtered_docs": [mock_docs[0]]  # Only good document
        }
        
        # Mock web search
        web_docs = [{"content": "Additional web result", "metadata": {"source": "web"}}]
        web_search.search.return_value = web_docs
        
        # Mock LLM response
        llm.generate.return_value = "Response based on filtered docs and web search"
        
        query = "Test query"
        result = crag_processor.process_query(query)
        
        assert result["response"] == "Response based on filtered docs and web search"
        assert len(result["sources"]) == 2  # Filtered doc + web doc
        assert result["evaluation"]["quality"] == "mixed"
        retriever.retrieve.assert_called_once_with(query)
        evaluator.evaluate_retrieval.assert_called_once()
        web_search.search.assert_called_once_with(query)
        llm.generate.assert_called_once()
    
    def test_process_query_retrieval_error(self, crag_processor, mock_components):
        """Test query processing when retrieval fails."""
        retriever, llm, evaluator, web_search = mock_components
        
        # Mock retrieval error
        retriever.retrieve.side_effect = RetrievalError("Retrieval failed")
        
        # Mock web search fallback
        web_docs = [{"content": "Web fallback result", "metadata": {"source": "web"}}]
        web_search.search.return_value = web_docs
        
        # Mock LLM response
        llm.generate.return_value = "Response from web fallback"
        
        query = "Test query"
        result = crag_processor.process_query(query)
        
        assert result["response"] == "Response from web fallback"
        assert result["sources"] == web_docs
        assert "error" in result
        retriever.retrieve.assert_called_once_with(query)
        web_search.search.assert_called_once_with(query)
        llm.generate.assert_called_once()
    
    def test_process_query_llm_error(self, crag_processor, mock_components):
        """Test query processing when LLM generation fails."""
        retriever, llm, evaluator, web_search = mock_components
        
        # Mock retrieval
        mock_docs = [{"content": "Test document", "metadata": {"source": "doc1.pdf"}}]
        retriever.retrieve.return_value = mock_docs
        
        # Mock evaluation
        evaluator.evaluate_retrieval.return_value = {
            "quality": "good",
            "relevance_score": 0.8,
            "confidence": 0.9
        }
        
        # Mock LLM error
        llm.generate.side_effect = LLMError("LLM generation failed")
        
        query = "Test query"
        
        with pytest.raises(LLMError):
            crag_processor.process_query(query)
    
    def test_process_query_empty_retrieval(self, crag_processor, mock_components):
        """Test query processing with empty retrieval results."""
        retriever, llm, evaluator, web_search = mock_components
        
        # Mock empty retrieval
        retriever.retrieve.return_value = []
        
        # Mock web search
        web_docs = [{"content": "Web result", "metadata": {"source": "web"}}]
        web_search.search.return_value = web_docs
        
        # Mock LLM response
        llm.generate.return_value = "Response from web search only"
        
        query = "Test query"
        result = crag_processor.process_query(query)
        
        assert result["response"] == "Response from web search only"
        assert result["sources"] == web_docs
        retriever.retrieve.assert_called_once_with(query)
        evaluator.evaluate_retrieval.assert_not_called()
        web_search.search.assert_called_once_with(query)
        llm.generate.assert_called_once()
