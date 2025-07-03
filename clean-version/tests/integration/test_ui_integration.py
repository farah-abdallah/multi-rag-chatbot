"""
Integration tests for the Streamlit UI components.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

from src.ui.streamlit_app import RAGChatbotApp
from src.ui.components import (
    DocumentUploader,
    ChatInterface,
    SourceHighlighter,
    SearchSettings
)


class TestStreamlitUIIntegration:
    """Integration tests for Streamlit UI components."""
    
    @pytest.fixture
    def mock_session_state(self):
        """Create mock Streamlit session state."""
        session_state = {
            "messages": [],
            "documents": [],
            "rag_processor": None,
            "search_settings": {
                "max_results": 5,
                "search_type": "similarity",
                "temperature": 0.7
            }
        }
        return session_state
    
    @pytest.fixture
    def mock_rag_processor(self):
        """Create mock RAG processor for testing."""
        mock_processor = Mock()
        mock_processor.process_query.return_value = {
            "response": "This is a test response about Python programming.",
            "sources": [
                {
                    "content": "Python is a high-level programming language.",
                    "metadata": {"source": "python_guide.pdf", "page": 1}
                },
                {
                    "content": "Python supports multiple programming paradigms.",
                    "metadata": {"source": "python_concepts.txt", "line": 15}
                }
            ],
            "evaluation": {
                "quality": "good",
                "relevance_score": 0.85,
                "confidence": 0.92
            }
        }
        return mock_processor
    
    @pytest.fixture
    def sample_uploaded_files(self):
        """Create sample uploaded files for testing."""
        temp_dir = tempfile.mkdtemp()
        
        files = []
        contents = [
            "Python is a versatile programming language used for web development, data analysis, and machine learning.",
            "Machine learning algorithms can automatically learn patterns from data without explicit programming.",
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers."
        ]
        
        for i, content in enumerate(contents):
            file_path = os.path.join(temp_dir, f"test_doc_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(content)
            files.append(file_path)
        
        yield files
        
        # Cleanup
        for file_path in files:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
    
    @patch('streamlit.session_state')
    def test_document_upload_integration(self, mock_session_state, sample_uploaded_files):
        """Test document upload integration."""
        # Mock session state
        mock_session_state.return_value = {"documents": [], "rag_processor": None}
        
        # Create document uploader
        uploader = DocumentUploader()
        
        # Mock file upload
        with patch('streamlit.file_uploader') as mock_file_uploader:
            # Mock uploaded files
            mock_files = []
            for file_path in sample_uploaded_files:
                with open(file_path, 'rb') as f:
                    mock_file = Mock()
                    mock_file.name = os.path.basename(file_path)
                    mock_file.read.return_value = f.read()
                    mock_files.append(mock_file)
            
            mock_file_uploader.return_value = mock_files
            
            # Test file processing
            with patch.object(uploader, 'process_uploaded_files') as mock_process:
                mock_process.return_value = [
                    {"content": "Document 1 content", "metadata": {"source": "test_doc_0.txt"}},
                    {"content": "Document 2 content", "metadata": {"source": "test_doc_1.txt"}},
                    {"content": "Document 3 content", "metadata": {"source": "test_doc_2.txt"}}
                ]
                
                documents = uploader.handle_file_upload()
                
                assert len(documents) == 3
                assert all("content" in doc for doc in documents)
                assert all("metadata" in doc for doc in documents)
    
    @patch('streamlit.session_state')
    @patch('streamlit.chat_message')
    @patch('streamlit.chat_input')
    def test_chat_interface_integration(self, mock_chat_input, mock_chat_message, mock_session_state, mock_rag_processor):
        """Test chat interface integration."""
        # Mock session state
        mock_session_state.return_value = {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."}
            ],
            "rag_processor": mock_rag_processor
        }
        
        # Mock user input
        mock_chat_input.return_value = "Tell me about machine learning"
        
        # Create chat interface
        chat_interface = ChatInterface()
        
        # Test message handling
        with patch.object(chat_interface, 'display_messages') as mock_display, \
             patch.object(chat_interface, 'handle_user_input') as mock_handle:
            
            chat_interface.render()
            
            mock_display.assert_called_once()
            mock_handle.assert_called_once_with("Tell me about machine learning")
    
    @patch('streamlit.session_state')
    def test_source_highlighter_integration(self, mock_session_state):
        """Test source highlighter integration."""
        # Mock session state with sources
        sources = [
            {
                "content": "Python is a high-level programming language known for its simplicity.",
                "metadata": {"source": "python_guide.pdf", "page": 1}
            },
            {
                "content": "Machine learning enables computers to learn from data automatically.",
                "metadata": {"source": "ml_basics.txt", "line": 10}
            }
        ]
        
        mock_session_state.return_value = {"current_sources": sources}
        
        # Create source highlighter
        highlighter = SourceHighlighter()
        
        # Test highlighting
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown:
            
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock()
            
            highlighter.display_sources(sources, query="Python programming")
            
            mock_expander.assert_called()
            mock_markdown.assert_called()
    
    @patch('streamlit.session_state')
    def test_search_settings_integration(self, mock_session_state):
        """Test search settings integration."""
        # Mock session state
        mock_session_state.return_value = {
            "search_settings": {
                "max_results": 5,
                "search_type": "similarity",
                "temperature": 0.7,
                "enable_web_search": True
            }
        }
        
        # Create search settings
        settings = SearchSettings()
        
        # Test settings rendering
        with patch('streamlit.sidebar') as mock_sidebar, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.checkbox') as mock_checkbox:
            
            mock_slider.return_value = 5
            mock_selectbox.return_value = "similarity"
            mock_checkbox.return_value = True
            
            settings.render()
            
            mock_slider.assert_called()
            mock_selectbox.assert_called()
            mock_checkbox.assert_called()
    
    @patch('streamlit.session_state')
    def test_full_app_integration(self, mock_session_state, mock_rag_processor):
        """Test full RAG chatbot app integration."""
        # Mock session state
        mock_session_state.return_value = {
            "messages": [],
            "documents": [],
            "rag_processor": mock_rag_processor,
            "search_settings": {
                "max_results": 5,
                "search_type": "similarity",
                "temperature": 0.7
            }
        }
        
        # Create app
        app = RAGChatbotApp()
        
        # Test app initialization
        with patch.object(app, 'setup_page') as mock_setup, \
             patch.object(app, 'render_sidebar') as mock_sidebar, \
             patch.object(app, 'render_main_content') as mock_main:
            
            app.run()
            
            mock_setup.assert_called_once()
            mock_sidebar.assert_called_once()
            mock_main.assert_called_once()
    
    @patch('streamlit.session_state')
    def test_query_processing_integration(self, mock_session_state, mock_rag_processor):
        """Test query processing integration."""
        # Mock session state
        mock_session_state.return_value = {
            "messages": [],
            "rag_processor": mock_rag_processor,
            "documents": [
                {"content": "Python content", "metadata": {"source": "python.txt"}}
            ]
        }
        
        # Create chat interface
        chat_interface = ChatInterface()
        
        # Test query processing
        user_query = "What is Python programming?"
        
        with patch('streamlit.chat_message') as mock_chat_message:
            mock_chat_message.return_value.__enter__ = Mock()
            mock_chat_message.return_value.__exit__ = Mock()
            
            response = chat_interface.process_query(user_query)
            
            assert response is not None
            mock_rag_processor.process_query.assert_called_once_with(user_query)
    
    @patch('streamlit.session_state')
    def test_error_handling_integration(self, mock_session_state):
        """Test error handling integration."""
        # Mock session state
        mock_session_state.return_value = {
            "messages": [],
            "rag_processor": None,
            "documents": []
        }
        
        # Create chat interface
        chat_interface = ChatInterface()
        
        # Test error handling when no RAG processor
        with patch('streamlit.error') as mock_error:
            response = chat_interface.process_query("Test query")
            
            assert response is None
            mock_error.assert_called_once()
    
    @patch('streamlit.session_state')
    def test_document_processing_integration(self, mock_session_state, sample_uploaded_files):
        """Test document processing integration."""
        # Mock session state
        mock_session_state.return_value = {
            "documents": [],
            "rag_processor": None
        }
        
        # Create document uploader
        uploader = DocumentUploader()
        
        # Test document processing
        with patch.object(uploader, 'load_documents') as mock_load, \
             patch.object(uploader, 'chunk_documents') as mock_chunk, \
             patch.object(uploader, 'create_vectorstore') as mock_vectorstore:
            
            mock_load.return_value = [
                {"content": "Document content", "metadata": {"source": "test.txt"}}
            ]
            mock_chunk.return_value = [
                {"content": "Chunk content", "metadata": {"source": "test.txt", "chunk_id": "1"}}
            ]
            mock_vectorstore.return_value = Mock()
            
            documents = uploader.process_documents(sample_uploaded_files)
            
            assert documents is not None
            mock_load.assert_called_once()
            mock_chunk.assert_called_once()
            mock_vectorstore.assert_called_once()
    
    @patch('streamlit.session_state')
    def test_real_time_updates_integration(self, mock_session_state, mock_rag_processor):
        """Test real-time updates integration."""
        # Mock session state with message history
        mock_session_state.return_value = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "rag_processor": mock_rag_processor
        }
        
        # Create chat interface
        chat_interface = ChatInterface()
        
        # Test adding new messages
        with patch('streamlit.chat_message') as mock_chat_message:
            mock_chat_message.return_value.__enter__ = Mock()
            mock_chat_message.return_value.__exit__ = Mock()
            
            chat_interface.add_message("user", "New user message")
            chat_interface.add_message("assistant", "New assistant response")
            
            # Verify messages were added
            assert len(mock_session_state.return_value["messages"]) == 4
    
    @patch('streamlit.session_state')
    def test_performance_monitoring_integration(self, mock_session_state, mock_rag_processor):
        """Test performance monitoring integration."""
        import time
        
        # Mock session state
        mock_session_state.return_value = {
            "messages": [],
            "rag_processor": mock_rag_processor,
            "performance_metrics": []
        }
        
        # Create chat interface
        chat_interface = ChatInterface()
        
        # Test performance tracking
        with patch('streamlit.sidebar') as mock_sidebar:
            start_time = time.time()
            
            chat_interface.process_query("Test query")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            assert processing_time > 0
            mock_rag_processor.process_query.assert_called_once()
    
    @patch('streamlit.session_state')
    def test_configuration_integration(self, mock_session_state):
        """Test configuration integration."""
        from config.settings import Settings
        
        # Mock session state
        mock_session_state.return_value = {
            "app_config": Settings()
        }
        
        # Create app
        app = RAGChatbotApp()
        
        # Test configuration loading
        with patch.object(app, 'load_config') as mock_load_config:
            mock_load_config.return_value = Settings()
            
            config = app.get_config()
            
            assert config is not None
            assert hasattr(config, 'CHUNK_SIZE')
            assert hasattr(config, 'MAX_RETRIEVAL_RESULTS')
            mock_load_config.assert_called_once()
