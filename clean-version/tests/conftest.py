"""
Test configuration and fixtures for Multi-RAG Chatbot tests.
"""
import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import json

# Test data
SAMPLE_DOCUMENT_CONTENT = """
This is a sample document for testing purposes. It contains information about artificial intelligence and machine learning.

Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines. Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.

Some key concepts in AI include:
- Natural Language Processing (NLP)
- Computer Vision
- Robotics
- Expert Systems

Machine learning algorithms can be categorized into:
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.
"""

SAMPLE_QUERY = "What is artificial intelligence?"

SAMPLE_ANSWER = """
Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks typically requiring human intelligence. According to the provided document, AI encompasses several key areas including Natural Language Processing (NLP), Computer Vision, Robotics, and Expert Systems. Machine learning is identified as a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.
"""


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return {
        'content': SAMPLE_DOCUMENT_CONTENT,
        'metadata': {
            'filename': 'test_document.txt',
            'filepath': '/tmp/test_document.txt',
            'file_size': len(SAMPLE_DOCUMENT_CONTENT),
            'file_extension': '.txt',
            'loaded_at': '2023-01-01T00:00:00',
            'content_length': len(SAMPLE_DOCUMENT_CONTENT),
            'document_hash': 'test_hash_123'
        },
        'id': 'test_doc_1'
    }


@pytest.fixture
def sample_documents():
    """Create multiple sample documents for testing."""
    return [
        {
            'content': SAMPLE_DOCUMENT_CONTENT,
            'metadata': {
                'filename': 'test_document_1.txt',
                'filepath': '/tmp/test_document_1.txt',
                'file_size': len(SAMPLE_DOCUMENT_CONTENT),
                'file_extension': '.txt',
                'loaded_at': '2023-01-01T00:00:00',
                'content_length': len(SAMPLE_DOCUMENT_CONTENT),
                'document_hash': 'test_hash_1'
            },
            'id': 'test_doc_1'
        },
        {
            'content': "This is another test document about machine learning and data science.",
            'metadata': {
                'filename': 'test_document_2.txt',
                'filepath': '/tmp/test_document_2.txt',
                'file_size': 65,
                'file_extension': '.txt',
                'loaded_at': '2023-01-01T00:00:00',
                'content_length': 65,
                'document_hash': 'test_hash_2'
            },
            'id': 'test_doc_2'
        }
    ]


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        {
            'content': "This is a sample document for testing purposes. It contains information about artificial intelligence and machine learning.",
            'metadata': {
                'chunk_id': 'test_doc_1_chunk_0',
                'document_id': 'test_doc_1',
                'chunk_index': 0,
                'total_chunks': 2,
                'start_position': 0,
                'end_position': 117,
                'chunk_size': 117,
                'overlap_size': 0,
                'created_at': '2023-01-01T00:00:00',
                'source_metadata': {
                    'filename': 'test_document.txt',
                    'filepath': '/tmp/test_document.txt'
                }
            },
            'similarity_score': 0.8,
            'relevance_score': 0.8
        },
        {
            'content': "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines.",
            'metadata': {
                'chunk_id': 'test_doc_1_chunk_1',
                'document_id': 'test_doc_1',
                'chunk_index': 1,
                'total_chunks': 2,
                'start_position': 100,
                'end_position': 200,
                'chunk_size': 100,
                'overlap_size': 0,
                'created_at': '2023-01-01T00:00:00',
                'source_metadata': {
                    'filename': 'test_document.txt',
                    'filepath': '/tmp/test_document.txt'
                }
            },
            'similarity_score': 0.9,
            'relevance_score': 0.9
        }
    ]


@pytest.fixture
def sample_web_results():
    """Create sample web search results for testing."""
    return [
        {
            'title': 'Introduction to Artificial Intelligence',
            'url': 'https://example.com/ai-intro',
            'content': 'Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.',
            'source': 'web',
            'timestamp': '2023-01-01T00:00:00',
            'relevance': 0.85,
            'metadata': {
                'domain': 'example.com',
                'content_length': 104,
                'processed_at': '2023-01-01T00:00:00'
            }
        },
        {
            'title': 'Machine Learning Basics',
            'url': 'https://example.com/ml-basics',
            'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.',
            'source': 'web',
            'timestamp': '2023-01-01T00:00:00',
            'relevance': 0.75,
            'metadata': {
                'domain': 'example.com',
                'content_length': 112,
                'processed_at': '2023-01-01T00:00:00'
            }
        }
    ]


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_DOCUMENT_CONTENT)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_api_keys():
    """Mock API keys for testing."""
    return ['test_key_1', 'test_key_2', 'test_key_3']


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'max_search_results': 5,
        'temperature': 0.1,
        'max_tokens': 2000,
        'use_web_search': True,
        'show_source_highlights': True
    }


@pytest.fixture
def sample_evaluation_metrics():
    """Sample evaluation metrics for testing."""
    return {
        'relevance_score': 0.85,
        'completeness_score': 0.80,
        'accuracy_score': 0.90,
        'confidence_score': 0.83,
        'overall_score': 0.85,
        'feedback': 'Good response with relevant information',
        'metadata': {
            'evaluation_timestamp': '2023-01-01T00:00:00',
            'num_sources': 3,
            'processing_time': 1.5
        }
    }


@pytest.fixture
def sample_knowledge_items():
    """Sample knowledge items for testing."""
    return [
        {
            'id': 'knowledge_1',
            'content': 'Artificial intelligence is a branch of computer science.',
            'keywords': ['artificial', 'intelligence', 'computer', 'science'],
            'category': 'technical',
            'confidence': 0.9,
            'sources': ['test_document.txt'],
            'created_at': '2023-01-01T00:00:00',
            'updated_at': '2023-01-01T00:00:00',
            'metadata': {
                'original_query': 'What is AI?',
                'context_count': 2
            }
        },
        {
            'id': 'knowledge_2',
            'content': 'Machine learning is a subset of artificial intelligence.',
            'keywords': ['machine', 'learning', 'artificial', 'intelligence'],
            'category': 'technical',
            'confidence': 0.85,
            'sources': ['test_document.txt'],
            'created_at': '2023-01-01T00:00:00',
            'updated_at': '2023-01-01T00:00:00',
            'metadata': {
                'original_query': 'What is machine learning?',
                'context_count': 1
            }
        }
    ]


@pytest.fixture
def sample_chat_history():
    """Sample chat history for testing."""
    return [
        {
            'role': 'user',
            'content': 'What is artificial intelligence?',
            'timestamp': '2023-01-01T00:00:00'
        },
        {
            'role': 'assistant',
            'content': SAMPLE_ANSWER,
            'timestamp': '2023-01-01T00:00:05',
            'metadata': {
                'confidence': 0.85,
                'sources_count': 2,
                'method_used': 'hybrid'
            }
        },
        {
            'role': 'user',
            'content': 'Tell me more about machine learning.',
            'timestamp': '2023-01-01T00:01:00'
        }
    ]


@pytest.fixture
def sample_highlights():
    """Sample source highlights for testing."""
    return [
        {
            'document_id': 'test_doc_1',
            'document_name': 'test_document.txt',
            'chunk_id': 'test_doc_1_chunk_0',
            'highlighted_text': 'Artificial intelligence (AI) is a branch of computer science',
            'context_before': 'This document discusses ',
            'context_after': ' that aims to create intelligent machines.',
            'relevance_score': 0.9,
            'page_number': 1,
            'section': 'Introduction',
            'metadata': {
                'filename': 'test_document.txt',
                'chunk_index': 0
            }
        }
    ]


def create_test_file(content: str, filename: str, directory: Path) -> Path:
    """Create a test file with given content."""
    file_path = directory / filename
    file_path.write_text(content)
    return file_path


def create_test_pdf(content: str, filename: str, directory: Path) -> Path:
    """Create a test PDF file (mock)."""
    # This would require actual PDF creation in a real implementation
    # For now, just create a text file with PDF extension
    file_path = directory / filename
    file_path.write_text(content)
    return file_path


def create_test_json(data: Dict[str, Any], filename: str, directory: Path) -> Path:
    """Create a test JSON file."""
    file_path = directory / filename
    file_path.write_text(json.dumps(data, indent=2))
    return file_path


def assert_document_structure(document: Dict[str, Any]):
    """Assert that a document has the expected structure."""
    assert 'content' in document
    assert 'metadata' in document
    assert 'id' in document
    assert isinstance(document['content'], str)
    assert isinstance(document['metadata'], dict)
    assert isinstance(document['id'], str)


def assert_chunk_structure(chunk: Dict[str, Any]):
    """Assert that a chunk has the expected structure."""
    assert 'content' in chunk
    assert 'metadata' in chunk
    assert isinstance(chunk['content'], str)
    assert isinstance(chunk['metadata'], dict)
    
    metadata = chunk['metadata']
    required_fields = ['chunk_id', 'document_id', 'chunk_index', 'total_chunks']
    for field in required_fields:
        assert field in metadata


def assert_evaluation_metrics(metrics: Dict[str, Any]):
    """Assert that evaluation metrics have the expected structure."""
    required_fields = ['relevance_score', 'completeness_score', 'accuracy_score', 'confidence_score', 'overall_score']
    for field in required_fields:
        assert field in metrics
        assert isinstance(metrics[field], (int, float))
        assert 0 <= metrics[field] <= 1


def assert_knowledge_item(item: Dict[str, Any]):
    """Assert that a knowledge item has the expected structure."""
    required_fields = ['id', 'content', 'keywords', 'category', 'confidence', 'sources']
    for field in required_fields:
        assert field in item
    
    assert isinstance(item['keywords'], list)
    assert isinstance(item['sources'], list)
    assert isinstance(item['confidence'], (int, float))
    assert 0 <= item['confidence'] <= 1


# Test constants
TEST_QUERY = "What is artificial intelligence?"
TEST_ANSWER = "Artificial intelligence is a branch of computer science."
TEST_DOCUMENT_CONTENT = SAMPLE_DOCUMENT_CONTENT
TEST_CHUNK_SIZE = 500
TEST_CHUNK_OVERLAP = 100
TEST_MAX_RESULTS = 10
TEST_SIMILARITY_THRESHOLD = 0.6
