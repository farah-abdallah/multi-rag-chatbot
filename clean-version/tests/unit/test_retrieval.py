"""
Unit tests for retrieval functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from src.core.retrieval import DocumentRetriever, EmbeddingRetriever
from src.utils.exceptions import RetrievalError


class TestDocumentRetriever:
    """Test cases for DocumentRetriever class."""
    
    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock vectorstore for testing."""
        mock_store = Mock()
        mock_store.similarity_search.return_value = [
            Mock(page_content="Document 1 content", metadata={"source": "doc1.pdf"}),
            Mock(page_content="Document 2 content", metadata={"source": "doc2.pdf"})
        ]
        return mock_store
    
    @pytest.fixture
    def retriever(self, mock_vectorstore):
        """Create DocumentRetriever instance with mock vectorstore."""
        return DocumentRetriever(vectorstore=mock_vectorstore)
    
    def test_init(self, retriever):
        """Test DocumentRetriever initialization."""
        assert retriever.vectorstore is not None
        assert retriever.k == 5  # default value
    
    def test_retrieve_success(self, retriever, mock_vectorstore):
        """Test successful document retrieval."""
        query = "test query"
        
        results = retriever.retrieve(query)
        
        assert len(results) == 2
        assert results[0]["content"] == "Document 1 content"
        assert results[0]["metadata"]["source"] == "doc1.pdf"
        assert results[1]["content"] == "Document 2 content"
        assert results[1]["metadata"]["source"] == "doc2.pdf"
        
        mock_vectorstore.similarity_search.assert_called_once_with(query, k=5)
    
    def test_retrieve_with_custom_k(self, mock_vectorstore):
        """Test retrieval with custom k value."""
        retriever = DocumentRetriever(vectorstore=mock_vectorstore, k=3)
        query = "test query"
        
        retriever.retrieve(query)
        
        mock_vectorstore.similarity_search.assert_called_once_with(query, k=3)
    
    def test_retrieve_empty_results(self, retriever, mock_vectorstore):
        """Test retrieval with empty results."""
        mock_vectorstore.similarity_search.return_value = []
        query = "test query"
        
        results = retriever.retrieve(query)
        
        assert results == []
        mock_vectorstore.similarity_search.assert_called_once_with(query, k=5)
    
    def test_retrieve_with_scores(self, retriever, mock_vectorstore):
        """Test retrieval with similarity scores."""
        # Mock similarity_search_with_score
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(page_content="Document 1", metadata={"source": "doc1.pdf"}), 0.9),
            (Mock(page_content="Document 2", metadata={"source": "doc2.pdf"}), 0.8)
        ]
        
        results = retriever.retrieve_with_scores("test query")
        
        assert len(results) == 2
        assert results[0]["content"] == "Document 1"
        assert results[0]["score"] == 0.9
        assert results[1]["content"] == "Document 2"
        assert results[1]["score"] == 0.8
    
    def test_retrieve_exception_handling(self, retriever, mock_vectorstore):
        """Test retrieval exception handling."""
        mock_vectorstore.similarity_search.side_effect = Exception("Vectorstore error")
        
        with pytest.raises(RetrievalError):
            retriever.retrieve("test query")


class TestEmbeddingRetriever:
    """Test cases for EmbeddingRetriever class."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model for testing."""
        mock_model = Mock()
        mock_model.embed_query.return_value = np.array([0.1, 0.2, 0.3])
        mock_model.embed_documents.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        return mock_model
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {"content": "Document 1 content", "metadata": {"source": "doc1.pdf"}},
            {"content": "Document 2 content", "metadata": {"source": "doc2.pdf"}}
        ]
    
    @pytest.fixture
    def embedding_retriever(self, mock_embedding_model):
        """Create EmbeddingRetriever instance with mock embedding model."""
        return EmbeddingRetriever(embedding_model=mock_embedding_model)
    
    def test_init(self, embedding_retriever):
        """Test EmbeddingRetriever initialization."""
        assert embedding_retriever.embedding_model is not None
        assert embedding_retriever.document_embeddings is None
        assert embedding_retriever.documents == []
    
    def test_add_documents(self, embedding_retriever, mock_embedding_model, sample_documents):
        """Test adding documents to the retriever."""
        embedding_retriever.add_documents(sample_documents)
        
        assert len(embedding_retriever.documents) == 2
        assert embedding_retriever.document_embeddings.shape == (2, 3)
        mock_embedding_model.embed_documents.assert_called_once_with(
            ["Document 1 content", "Document 2 content"]
        )
    
    def test_retrieve_similarity(self, embedding_retriever, mock_embedding_model, sample_documents):
        """Test similarity-based retrieval."""
        # Add documents first
        embedding_retriever.add_documents(sample_documents)
        
        # Mock similarity calculation
        with patch('numpy.dot') as mock_dot:
            mock_dot.return_value = np.array([0.8, 0.6])  # similarity scores
            
            results = embedding_retriever.retrieve("test query", k=2)
            
            assert len(results) == 2
            assert results[0]["content"] == "Document 1 content"  # Higher similarity
            assert results[1]["content"] == "Document 2 content"
            mock_embedding_model.embed_query.assert_called_once_with("test query")
    
    def test_retrieve_with_threshold(self, embedding_retriever, mock_embedding_model, sample_documents):
        """Test retrieval with similarity threshold."""
        embedding_retriever.add_documents(sample_documents)
        
        with patch('numpy.dot') as mock_dot:
            mock_dot.return_value = np.array([0.8, 0.3])  # One below threshold
            
            results = embedding_retriever.retrieve("test query", k=2, threshold=0.5)
            
            assert len(results) == 1
            assert results[0]["content"] == "Document 1 content"
    
    def test_retrieve_empty_documents(self, embedding_retriever):
        """Test retrieval with no documents added."""
        results = embedding_retriever.retrieve("test query")
        
        assert results == []
    
    def test_retrieve_exception_handling(self, embedding_retriever, mock_embedding_model):
        """Test retrieval exception handling."""
        mock_embedding_model.embed_query.side_effect = Exception("Embedding error")
        
        with pytest.raises(RetrievalError):
            embedding_retriever.retrieve("test query")
    
    def test_update_documents(self, embedding_retriever, mock_embedding_model, sample_documents):
        """Test updating documents in the retriever."""
        # Add initial documents
        embedding_retriever.add_documents(sample_documents)
        
        # Add more documents
        new_documents = [
            {"content": "Document 3 content", "metadata": {"source": "doc3.pdf"}}
        ]
        embedding_retriever.add_documents(new_documents)
        
        assert len(embedding_retriever.documents) == 3
        assert embedding_retriever.document_embeddings.shape == (3, 3)
    
    def test_clear_documents(self, embedding_retriever, sample_documents):
        """Test clearing documents from the retriever."""
        embedding_retriever.add_documents(sample_documents)
        embedding_retriever.clear_documents()
        
        assert len(embedding_retriever.documents) == 0
        assert embedding_retriever.document_embeddings is None
