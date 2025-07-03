"""
Unit tests for document chunking functionality.
"""

import pytest
from unittest.mock import Mock, patch
from src.document.chunking import (
    TextChunker, 
    SemanticChunker, 
    RecursiveChunker,
    chunk_documents
)
from src.utils.exceptions import ChunkingError


class TestTextChunker:
    """Test cases for TextChunker class."""
    
    @pytest.fixture
    def text_chunker(self):
        """Create TextChunker instance for testing."""
        return TextChunker(chunk_size=100, chunk_overlap=20)
    
    def test_init(self, text_chunker):
        """Test TextChunker initialization."""
        assert text_chunker.chunk_size == 100
        assert text_chunker.chunk_overlap == 20
    
    def test_chunk_text_simple(self, text_chunker):
        """Test simple text chunking."""
        text = "This is a simple text that should be chunked into smaller pieces for testing."
        
        chunks = text_chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        assert all(len(chunk) <= 100 for chunk in chunks)
    
    def test_chunk_text_with_overlap(self):
        """Test text chunking with overlap."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a longer text that will definitely need to be split into multiple chunks to test the overlap functionality properly."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        # Check that there's overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            # Some words from the end of current chunk should appear in next chunk
            current_words = chunks[i].split()
            next_words = chunks[i + 1].split()
            overlap_found = any(word in next_words for word in current_words[-3:])
            assert overlap_found or len(current_words) < 3
    
    def test_chunk_text_empty(self, text_chunker):
        """Test chunking empty text."""
        chunks = text_chunker.chunk_text("")
        assert chunks == []
    
    def test_chunk_text_shorter_than_chunk_size(self, text_chunker):
        """Test chunking text shorter than chunk size."""
        text = "Short text."
        chunks = text_chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_documents(self, text_chunker):
        """Test chunking multiple documents."""
        documents = [
            {"content": "First document content", "metadata": {"source": "doc1.txt"}},
            {"content": "Second document content", "metadata": {"source": "doc2.txt"}}
        ]
        
        chunked_docs = text_chunker.chunk_documents(documents)
        
        assert len(chunked_docs) == 2
        assert all("chunk_id" in doc["metadata"] for doc in chunked_docs)
        assert all("chunk_index" in doc["metadata"] for doc in chunked_docs)
        assert chunked_docs[0]["metadata"]["source"] == "doc1.txt"
        assert chunked_docs[1]["metadata"]["source"] == "doc2.txt"


class TestSemanticChunker:
    """Test cases for SemanticChunker class."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model for testing."""
        mock_model = Mock()
        mock_model.embed_documents.return_value = [
            [0.1, 0.2, 0.3],  # Sentence 1
            [0.1, 0.2, 0.35], # Sentence 2 (similar to 1)
            [0.8, 0.9, 0.1],  # Sentence 3 (different)
            [0.8, 0.85, 0.15] # Sentence 4 (similar to 3)
        ]
        return mock_model
    
    @pytest.fixture
    def semantic_chunker(self, mock_embedding_model):
        """Create SemanticChunker instance for testing."""
        return SemanticChunker(
            embedding_model=mock_embedding_model,
            similarity_threshold=0.8
        )
    
    def test_init(self, semantic_chunker):
        """Test SemanticChunker initialization."""
        assert semantic_chunker.embedding_model is not None
        assert semantic_chunker.similarity_threshold == 0.8
    
    @patch('nltk.sent_tokenize')
    def test_chunk_text_semantic(self, mock_sent_tokenize, semantic_chunker):
        """Test semantic text chunking."""
        # Mock sentence tokenization
        sentences = [
            "First sentence about topic A.",
            "Second sentence also about topic A.",
            "Third sentence about topic B.",
            "Fourth sentence also about topic B."
        ]
        mock_sent_tokenize.return_value = sentences
        
        text = " ".join(sentences)
        chunks = semantic_chunker.chunk_text(text)
        
        # Should group similar sentences together
        assert len(chunks) >= 1
        assert all(len(chunk.strip()) > 0 for chunk in chunks)
    
    def test_chunk_text_single_sentence(self, semantic_chunker):
        """Test semantic chunking with single sentence."""
        text = "This is a single sentence."
        
        with patch('nltk.sent_tokenize', return_value=[text]):
            chunks = semantic_chunker.chunk_text(text)
            
            assert len(chunks) == 1
            assert chunks[0].strip() == text
    
    def test_chunk_text_empty(self, semantic_chunker):
        """Test semantic chunking with empty text."""
        chunks = semantic_chunker.chunk_text("")
        assert chunks == []
    
    def test_calculate_similarity(self, semantic_chunker):
        """Test similarity calculation."""
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.1, 0.2, 0.35]
        
        similarity = semantic_chunker._calculate_similarity(embedding1, embedding2)
        
        assert 0 <= similarity <= 1
        assert similarity > 0.9  # Should be high similarity


class TestRecursiveChunker:
    """Test cases for RecursiveChunker class."""
    
    @pytest.fixture
    def recursive_chunker(self):
        """Create RecursiveChunker instance for testing."""
        return RecursiveChunker(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", ".", " "]
        )
    
    def test_init(self, recursive_chunker):
        """Test RecursiveChunker initialization."""
        assert recursive_chunker.chunk_size == 100
        assert recursive_chunker.chunk_overlap == 20
        assert recursive_chunker.separators == ["\n\n", "\n", ".", " "]
    
    def test_chunk_text_with_paragraphs(self, recursive_chunker):
        """Test recursive chunking with paragraph breaks."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        chunks = recursive_chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        assert all(len(chunk) <= 100 for chunk in chunks)
    
    def test_chunk_text_with_sentences(self, recursive_chunker):
        """Test recursive chunking with sentence breaks."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = recursive_chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        assert all(len(chunk) <= 100 for chunk in chunks)
    
    def test_chunk_text_long_without_separators(self, recursive_chunker):
        """Test recursive chunking with long text without separators."""
        text = "a" * 200  # Long text without separators
        
        chunks = recursive_chunker.chunk_text(text)
        
        assert len(chunks) >= 2
        assert all(len(chunk) <= 100 for chunk in chunks)
    
    def test_chunk_text_empty(self, recursive_chunker):
        """Test recursive chunking with empty text."""
        chunks = recursive_chunker.chunk_text("")
        assert chunks == []
    
    def test_find_best_split(self, recursive_chunker):
        """Test finding best split point."""
        text = "First sentence. Second sentence. Third sentence."
        
        split_point = recursive_chunker._find_best_split(text, 30)
        
        assert 0 < split_point < len(text)
    
    def test_split_with_overlap(self, recursive_chunker):
        """Test splitting text with overlap."""
        text = "First sentence. Second sentence. Third sentence."
        
        chunks = recursive_chunker._split_with_overlap(text, 25)
        
        assert len(chunks) >= 2
        assert all(len(chunk) <= 25 for chunk in chunks)


class TestChunkingUtilities:
    """Test cases for chunking utility functions."""
    
    def test_chunk_documents_with_different_chunkers(self):
        """Test chunking documents with different chunker types."""
        documents = [
            {"content": "First document content", "metadata": {"source": "doc1.txt"}},
            {"content": "Second document content", "metadata": {"source": "doc2.txt"}}
        ]
        
        # Test with text chunker
        text_chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        chunked_docs = chunk_documents(documents, text_chunker)
        
        assert len(chunked_docs) == 2
        assert all("chunk_id" in doc["metadata"] for doc in chunked_docs)
    
    def test_chunk_documents_empty_list(self):
        """Test chunking empty document list."""
        documents = []
        text_chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        chunked_docs = chunk_documents(documents, text_chunker)
        
        assert chunked_docs == []
    
    def test_chunk_documents_with_error(self):
        """Test chunking documents with error."""
        documents = [
            {"content": "Test content", "metadata": {"source": "doc1.txt"}}
        ]
        
        # Mock chunker that raises error
        mock_chunker = Mock()
        mock_chunker.chunk_documents.side_effect = Exception("Chunking error")
        
        with pytest.raises(ChunkingError):
            chunk_documents(documents, mock_chunker)
    
    def test_estimate_chunk_count(self):
        """Test estimating chunk count."""
        from src.document.chunking import estimate_chunk_count
        
        text = "This is a test text. " * 100  # Repeat to make it long
        
        count = estimate_chunk_count(text, chunk_size=100, chunk_overlap=20)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_merge_small_chunks(self):
        """Test merging small chunks."""
        from src.document.chunking import merge_small_chunks
        
        chunks = [
            {"content": "Small chunk 1", "metadata": {"source": "doc1.txt"}},
            {"content": "Small chunk 2", "metadata": {"source": "doc1.txt"}},
            {"content": "This is a much longer chunk that should not be merged", "metadata": {"source": "doc1.txt"}}
        ]
        
        merged = merge_small_chunks(chunks, min_chunk_size=30)
        
        assert len(merged) < len(chunks)
        assert any(len(chunk["content"]) >= 30 for chunk in merged)
