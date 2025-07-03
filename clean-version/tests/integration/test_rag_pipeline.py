"""
Integration tests for the complete RAG pipeline.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.crag import CRAGProcessor
from src.core.retrieval import DocumentRetriever
from src.core.evaluation import RetrievalEvaluator
from src.document.loader import load_documents
from src.document.chunking import TextChunker
from src.llm.gemini import GeminiLLM
from src.llm.api_manager import APIKeyManager
from src.search.web import DuckDuckGoSearcher
from src.utils.exceptions import RetrievalError, LLMError, SearchError


class TestEndToEndRAGPipeline:
    """Integration tests for the complete RAG pipeline."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"source": "python_intro.txt", "type": "text"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "metadata": {"source": "ml_basics.txt", "type": "text"}
            },
            {
                "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "metadata": {"source": "deep_learning.txt", "type": "text"}
            }
        ]
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create sample text files
        files = []
        contents = [
            "Python is a versatile programming language.",
            "Machine learning algorithms can solve complex problems.",
            "Data science combines statistics and programming."
        ]
        
        for i, content in enumerate(contents):
            file_path = os.path.join(temp_dir, f"sample_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(content)
            files.append(file_path)
        
        yield files
        
        # Cleanup
        for file_path in files:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for integration testing."""
        # Mock vectorstore
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [
            Mock(page_content="Python programming content", metadata={"source": "python.txt"}),
            Mock(page_content="Machine learning content", metadata={"source": "ml.txt"})
        ]
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm.generate.return_value = "Generated response based on retrieved documents"
        
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_retrieval.return_value = {
            "quality": "good",
            "relevance_score": 0.8,
            "confidence": 0.9
        }
        
        # Mock web searcher
        mock_web_searcher = Mock()
        mock_web_searcher.search.return_value = [
            {"content": "Web search result", "metadata": {"source": "web", "title": "Web Result"}}
        ]
        
        return {
            "vectorstore": mock_vectorstore,
            "llm": mock_llm,
            "evaluator": mock_evaluator,
            "web_searcher": mock_web_searcher
        }
    
    def test_document_loading_and_chunking_pipeline(self, temp_files):
        """Test the complete document loading and chunking pipeline."""
        # Load documents
        documents = load_documents(temp_files)
        
        assert len(documents) == 3
        assert all("content" in doc for doc in documents)
        assert all("metadata" in doc for doc in documents)
        
        # Chunk documents
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        chunked_docs = chunker.chunk_documents(documents)
        
        assert len(chunked_docs) >= 3  # At least one chunk per document
        assert all("chunk_id" in doc["metadata"] for doc in chunked_docs)
        assert all("chunk_index" in doc["metadata"] for doc in chunked_docs)
    
    def test_retrieval_pipeline(self, mock_components, sample_documents):
        """Test the retrieval pipeline with mock components."""
        # Create retriever
        retriever = DocumentRetriever(
            vectorstore=mock_components["vectorstore"],
            k=2
        )
        
        # Test retrieval
        query = "Tell me about Python programming"
        results = retriever.retrieve(query)
        
        assert len(results) == 2
        assert all("content" in result for result in results)
        assert all("metadata" in result for result in results)
        mock_components["vectorstore"].similarity_search.assert_called_once_with(query, k=2)
    
    def test_crag_pipeline_with_good_retrieval(self, mock_components):
        """Test CRAG pipeline with good retrieval results."""
        # Create CRAG processor
        crag = CRAGProcessor(
            retriever=DocumentRetriever(mock_components["vectorstore"]),
            llm=mock_components["llm"],
            evaluator=mock_components["evaluator"],
            web_search=mock_components["web_searcher"]
        )
        
        # Process query
        query = "What is Python programming?"
        result = crag.process_query(query)
        
        assert "response" in result
        assert "sources" in result
        assert "evaluation" in result
        assert result["response"] == "Generated response based on retrieved documents"
        assert result["evaluation"]["quality"] == "good"
        
        # Verify component interactions
        mock_components["vectorstore"].similarity_search.assert_called_once()
        mock_components["evaluator"].evaluate_retrieval.assert_called_once()
        mock_components["llm"].generate.assert_called_once()
        mock_components["web_searcher"].search.assert_not_called()  # Not needed for good retrieval
    
    def test_crag_pipeline_with_poor_retrieval(self, mock_components):
        """Test CRAG pipeline with poor retrieval that triggers web search."""
        # Configure evaluator to return poor quality
        mock_components["evaluator"].evaluate_retrieval.return_value = {
            "quality": "poor",
            "relevance_score": 0.2,
            "confidence": 0.3
        }
        
        # Create CRAG processor
        crag = CRAGProcessor(
            retriever=DocumentRetriever(mock_components["vectorstore"]),
            llm=mock_components["llm"],
            evaluator=mock_components["evaluator"],
            web_search=mock_components["web_searcher"]
        )
        
        # Process query
        query = "What is quantum computing?"
        result = crag.process_query(query)
        
        assert "response" in result
        assert "sources" in result
        assert "evaluation" in result
        assert result["evaluation"]["quality"] == "poor"
        
        # Verify web search was triggered
        mock_components["web_searcher"].search.assert_called_once_with(query)
    
    def test_api_key_management_integration(self):
        """Test API key management integration."""
        api_keys = ["key1", "key2", "key3"]
        api_manager = APIKeyManager(api_keys)
        
        # Test key rotation and statistics
        for i in range(5):
            key = api_manager.get_current_key()
            if i % 2 == 0:
                api_manager.mark_key_success(key)
            else:
                api_manager.mark_key_failure(key)
            api_manager.rotate_key()
        
        # Check statistics
        all_stats = api_manager.get_all_stats()
        assert len(all_stats) == 3
        assert all(stats["requests"] > 0 for stats in all_stats.values())
        
        # Test best key selection
        best_key = api_manager.get_best_key()
        assert best_key in api_keys
    
    @patch('src.search.web.duckduckgo_search.DDGS')
    def test_web_search_integration(self, mock_ddgs):
        """Test web search integration."""
        # Mock DuckDuckGo response
        mock_ddgs_instance = Mock()
        mock_ddgs.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = [
            {"title": "Python Tutorial", "href": "http://python.org", "body": "Learn Python programming"},
            {"title": "ML Guide", "href": "http://ml.com", "body": "Machine learning basics"}
        ]
        
        # Test web search
        searcher = DuckDuckGoSearcher(max_results=2)
        results = searcher.search("Python programming tutorial")
        
        assert len(results) == 2
        assert all("content" in result for result in results)
        assert all("metadata" in result for result in results)
        assert "Python Tutorial" in results[0]["content"]
        assert "ML Guide" in results[1]["content"]
    
    def test_error_handling_integration(self, mock_components):
        """Test error handling across the pipeline."""
        # Test retrieval error
        mock_components["vectorstore"].similarity_search.side_effect = Exception("Retrieval failed")
        
        retriever = DocumentRetriever(mock_components["vectorstore"])
        
        with pytest.raises(RetrievalError):
            retriever.retrieve("test query")
    
    def test_multilingual_support_integration(self, mock_components):
        """Test multilingual support integration."""
        # Mock multilingual content
        mock_components["vectorstore"].similarity_search.return_value = [
            Mock(page_content="Python es un lenguaje de programación", metadata={"source": "python_es.txt"}),
            Mock(page_content="机器学习是人工智能的一个分支", metadata={"source": "ml_zh.txt"})
        ]
        
        retriever = DocumentRetriever(mock_components["vectorstore"])
        results = retriever.retrieve("programación Python")
        
        assert len(results) == 2
        assert "Python es un lenguaje" in results[0]["content"]
        assert "机器学习" in results[1]["content"]
    
    def test_performance_monitoring_integration(self, mock_components):
        """Test performance monitoring integration."""
        import time
        
        # Create CRAG processor with timing
        crag = CRAGProcessor(
            retriever=DocumentRetriever(mock_components["vectorstore"]),
            llm=mock_components["llm"],
            evaluator=mock_components["evaluator"],
            web_search=mock_components["web_searcher"]
        )
        
        # Process multiple queries and measure timing
        queries = [
            "What is Python?",
            "How does machine learning work?",
            "Explain deep learning algorithms"
        ]
        
        results = []
        for query in queries:
            start_time = time.time()
            result = crag.process_query(query)
            end_time = time.time()
            
            result["processing_time"] = end_time - start_time
            results.append(result)
        
        assert len(results) == 3
        assert all("processing_time" in result for result in results)
        assert all(result["processing_time"] > 0 for result in results)
    
    def test_concurrent_processing_integration(self, mock_components):
        """Test concurrent processing capabilities."""
        import threading
        import time
        
        # Create CRAG processor
        crag = CRAGProcessor(
            retriever=DocumentRetriever(mock_components["vectorstore"]),
            llm=mock_components["llm"],
            evaluator=mock_components["evaluator"],
            web_search=mock_components["web_searcher"]
        )
        
        # Test concurrent queries
        queries = [
            "What is Python?",
            "How does ML work?",
            "Explain AI concepts"
        ]
        
        results = []
        threads = []
        
        def process_query(query):
            result = crag.process_query(query)
            results.append(result)
        
        # Start threads
        for query in queries:
            thread = threading.Thread(target=process_query, args=(query,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        assert len(results) == 3
        assert all("response" in result for result in results)
    
    def test_configuration_integration(self):
        """Test configuration integration."""
        from config.settings import Settings
        
        # Test loading configuration
        settings = Settings()
        
        # Verify essential settings exist
        assert hasattr(settings, 'CHUNK_SIZE')
        assert hasattr(settings, 'CHUNK_OVERLAP')
        assert hasattr(settings, 'MAX_RETRIEVAL_RESULTS')
        assert hasattr(settings, 'LLM_TEMPERATURE')
        
        # Test configuration validation
        assert settings.CHUNK_SIZE > 0
        assert settings.CHUNK_OVERLAP >= 0
        assert settings.MAX_RETRIEVAL_RESULTS > 0
        assert 0 <= settings.LLM_TEMPERATURE <= 1
