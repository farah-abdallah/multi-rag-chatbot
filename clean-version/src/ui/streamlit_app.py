"""
Main Streamlit application for Multi-RAG Chatbot.
"""
import streamlit as st
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os

# Import all components
from ..core.crag import CRAGProcessor
from ..core.retrieval import HybridRetriever, RetrievalQuery
from ..core.evaluation import EvaluationSystem
from ..core.knowledge import KnowledgeRefinementSystem, KnowledgeBase
from ..document.loader import DocumentProcessor
from ..document.chunking import DocumentChunker
from ..document.viewer import DocumentViewer
from ..llm.gemini import GeminiLLM
from ..search.web import WebSearcher
from ..utils.logging import get_logger
from ..utils.exceptions import MultiRAGError
from .components import (
    render_sidebar, render_file_uploader, render_chat_interface,
    render_source_highlights, render_system_stats, render_settings
)
from .styles import apply_custom_styles
from config.settings import settings

# Configure logging
logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title=settings.page_title,
    page_icon=settings.page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles
apply_custom_styles()


class MultiRAGChatbot:
    """Main Multi-RAG Chatbot application."""
    
    def __init__(self):
        self.initialize_session_state()
        
        # Initialize component attributes with default values
        self.llm = None
        self.retriever = None
        self.web_searcher = None
        self.crag_processor = None
        self.evaluation_system = None
        self.knowledge_base = None
        self.knowledge_system = None
        self.document_processor = None
        self.document_chunker = None
        self.document_viewer = None
        
        self.initialize_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.documents = []
            st.session_state.chat_history = []
            st.session_state.system_ready = False
            st.session_state.processing = False
            st.session_state.last_response = None
            st.session_state.source_highlights = []
            st.session_state.knowledge_base = None
            st.session_state.settings = {
                'use_web_search': True,
                'chunk_size': settings.chunk_size,
                'chunk_overlap': settings.chunk_overlap,
                'max_search_results': settings.max_search_results,
                'show_source_highlights': True,
                'show_system_stats': False
            }
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            with st.spinner("Initializing Multi-RAG Chatbot..."):
                # Initialize LLM
                if self.llm is None:
                    self.llm = GeminiLLM()
                
                # Initialize retrieval system
                if self.retriever is None:
                    self.retriever = HybridRetriever()
                    self.retriever.initialize()
                
                # Initialize web searcher
                if self.web_searcher is None:
                    self.web_searcher = WebSearcher()
                
                # Initialize CRAG processor
                if self.crag_processor is None:
                    self.crag_processor = CRAGProcessor(self.llm, self.web_searcher)
                
                # Initialize evaluation system
                if self.evaluation_system is None:
                    self.evaluation_system = EvaluationSystem(self.llm)
                
                # Initialize knowledge system
                if self.knowledge_base is None:
                    self.knowledge_base = KnowledgeBase()
                if self.knowledge_system is None:
                    self.knowledge_system = KnowledgeRefinementSystem(self.llm, self.knowledge_base)
                
                # Initialize document processing
                if self.document_processor is None:
                    self.document_processor = DocumentProcessor()
                if self.document_chunker is None:
                    self.document_chunker = DocumentChunker(
                        chunk_size=st.session_state.settings['chunk_size'],
                        overlap_size=st.session_state.settings['chunk_overlap']
                    )
                
                # Initialize document viewer
                if self.document_viewer is None:
                    self.document_viewer = DocumentViewer()
                
                st.session_state.initialized = True
                st.session_state.system_ready = True
                
                logger.info("Multi-RAG Chatbot initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            st.error(f"Failed to initialize system: {str(e)}")
            st.session_state.system_ready = False
    
    def run(self):
        """Run the main application."""
        st.title("🤖 Multi-RAG Chatbot")
        st.markdown("---")
        
        if not st.session_state.system_ready:
            st.error("System initialization failed. Please refresh the page.")
            return
        
        # Render sidebar
        sidebar_action = render_sidebar()
        
        # Handle sidebar actions
        if sidebar_action == "upload_documents":
            self.handle_document_upload()
        elif sidebar_action == "clear_documents":
            self.handle_clear_documents()
        elif sidebar_action == "show_settings":
            self.show_settings()
        elif sidebar_action == "show_stats":
            self.show_system_stats()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chat interface
            self.render_chat_interface()
        
        with col2:
            # Source highlights and additional info
            self.render_right_panel()
    
    def handle_document_upload(self):
        """Handle document upload."""
        st.subheader("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'pptx', 'txt', 'md', 'csv', 'xlsx', 'xls', 'json'],
            help="Upload documents to add to the knowledge base"
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                try:
                    with st.spinner("Processing documents..."):
                        # Convert uploaded files to the expected format
                        file_data = []
                        for file in uploaded_files:
                            file_data.append({
                                'name': file.name,
                                'content': file.read()
                            })
                        
                        # Process documents
                        documents = self.document_processor.process_uploaded_files(file_data)
                        
                        # Chunk documents
                        chunks = self.document_chunker.chunk_documents(documents)
                        
                        # Add to retrieval system
                        chunk_dicts = [chunk.to_dict() for chunk in chunks]
                        self.retriever.add_documents(chunk_dicts)
                        
                        # Add to document viewer
                        for doc in documents:
                            self.document_viewer.add_document(doc)
                        
                        for chunk in chunks:
                            self.document_viewer.add_chunk(chunk.to_dict())
                        
                        # Update session state
                        st.session_state.documents.extend(documents)
                        
                        st.success(f"Successfully processed {len(documents)} documents with {len(chunks)} chunks")
                        logger.info(f"Processed {len(documents)} documents")
                        
                except Exception as e:
                    logger.error(f"Error processing documents: {str(e)}")
                    st.error(f"Error processing documents: {str(e)}")
    
    def handle_clear_documents(self):
        """Handle clearing all documents."""
        if st.button("Clear All Documents", type="primary"):
            try:
                # Clear all systems
                if self.retriever:
                    self.retriever.clear()
                if self.document_processor:
                    self.document_processor.clear_processed_documents()
                
                # Clear session state
                st.session_state.documents = []
                st.session_state.chat_history = []
                st.session_state.source_highlights = []
                
                st.success("All documents cleared")
                logger.info("All documents cleared")
                
            except Exception as e:
                logger.error(f"Error clearing documents: {str(e)}")
                st.error(f"Error clearing documents: {str(e)}")
    
    def show_settings(self):
        """Show system settings."""
        st.subheader("⚙️ Settings")
        
        settings_changed = False
        
        # Web search settings
        use_web_search = st.checkbox(
            "Enable Web Search",
            value=st.session_state.settings['use_web_search'],
            help="Enable web search to supplement document knowledge"
        )
        
        if use_web_search != st.session_state.settings['use_web_search']:
            st.session_state.settings['use_web_search'] = use_web_search
            settings_changed = True
        
        # Chunking settings
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=st.session_state.settings['chunk_size'],
            step=100,
            help="Size of document chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=st.session_state.settings['chunk_overlap'],
            step=50,
            help="Overlap between adjacent chunks"
        )
        
        if chunk_size != st.session_state.settings['chunk_size']:
            st.session_state.settings['chunk_size'] = chunk_size
            settings_changed = True
        
        if chunk_overlap != st.session_state.settings['chunk_overlap']:
            st.session_state.settings['chunk_overlap'] = chunk_overlap
            settings_changed = True
        
        # Display settings
        show_highlights = st.checkbox(
            "Show Source Highlights",
            value=st.session_state.settings['show_source_highlights'],
            help="Show highlighted sources in responses"
        )
        
        show_stats = st.checkbox(
            "Show System Statistics",
            value=st.session_state.settings['show_system_stats'],
            help="Show system performance statistics"
        )
        
        if show_highlights != st.session_state.settings['show_source_highlights']:
            st.session_state.settings['show_source_highlights'] = show_highlights
            settings_changed = True
        
        if show_stats != st.session_state.settings['show_system_stats']:
            st.session_state.settings['show_system_stats'] = show_stats
            settings_changed = True
        
        if settings_changed:
            st.success("Settings updated")
            logger.info("Settings updated")
    
    def show_system_stats(self):
        """Show system statistics."""
        st.subheader("📊 System Statistics")
        
        try:
            # Get retrieval stats
            retrieval_stats = self.retriever.get_stats() if self.retriever else {}
            
            # Get document processing stats
            processing_stats = self.document_processor.get_processing_stats() if self.document_processor else {}
            
            # Get LLM stats
            llm_stats = self.llm.get_api_stats() if self.llm else {}
            
            # Get evaluation stats
            evaluation_stats = self.evaluation_system.get_evaluation_stats() if self.evaluation_system else {}
            
            # Display stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Documents Processed", processing_stats.get('total_documents', 0))
                st.metric("Total Chunks", retrieval_stats.get('total_documents', 0))
                st.metric("API Calls", llm_stats.get('total_usage', 0))
            
            with col2:
                st.metric("Available API Keys", llm_stats.get('available_keys', 0))
                st.metric("Total Evaluations", evaluation_stats.get('total_evaluations', 0))
                st.metric("Average Quality Score", f"{evaluation_stats.get('average_scores', {}).get('overall', 0):.2f}")
            
            # Detailed stats
            with st.expander("Detailed Statistics"):
                st.json({
                    'retrieval': retrieval_stats,
                    'processing': processing_stats,
                    'llm': llm_stats,
                    'evaluation': evaluation_stats
                })
        
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            st.error(f"Error getting system stats: {str(e)}")
    
    def render_chat_interface(self):
        """Render the chat interface."""
        st.subheader("💬 Chat")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")
                
                # Show source highlights if available
                if (message['role'] == 'assistant' and 
                    st.session_state.settings['show_source_highlights'] and 
                    'highlights' in message):
                    with st.expander("Source Highlights"):
                        for highlight in message['highlights']:
                            st.markdown(f"**{highlight.document_name}** (Score: {highlight.relevance_score:.2f})")
                            st.markdown(f"*{highlight.highlighted_text}*")
                            st.markdown("---")
        
        # Chat input
        query = st.text_input("Ask a question:", placeholder="Enter your question here...")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Send", disabled=st.session_state.processing):
                if query.strip():
                    self.handle_query(query)
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.source_highlights = []
                st.rerun()
    
    def handle_query(self, query: str):
        """Handle user query."""
        try:
            st.session_state.processing = True
            
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': query,
                'timestamp': datetime.now().isoformat()
            })
            
            with st.spinner("Processing your question..."):
                # Get response using asyncio
                response_data = asyncio.run(self.process_query_async(query))
                
                # Add assistant response to chat history
                assistant_message = {
                    'role': 'assistant',
                    'content': response_data['answer'],
                    'timestamp': datetime.now().isoformat(),
                    'highlights': response_data.get('highlights', []),
                    'metadata': response_data.get('metadata', {})
                }
                
                st.session_state.chat_history.append(assistant_message)
                st.session_state.source_highlights = response_data.get('highlights', [])
                
                logger.info(f"Processed query: {query[:100]}...")
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            st.error(f"Error processing query: {str(e)}")
        
        finally:
            st.session_state.processing = False
            st.rerun()
    
    async def process_query_async(self, query: str) -> Dict[str, Any]:
        """Process query asynchronously."""
        try:
            # Check if components are initialized
            if not self.retriever or not self.crag_processor:
                return {
                    'answer': "System not fully initialized. Please refresh the page.",
                    'highlights': [],
                    'metadata': {'error': 'System not initialized'}
                }
            
            # Retrieve relevant documents
            retrieval_query = RetrievalQuery(query=query, max_results=10)
            retrieved_docs = self.retriever.retrieve(retrieval_query)
            
            # Use CRAG for enhanced processing
            crag_result = await self.crag_processor.process_query(
                query,
                retrieved_docs,
                use_web_search=st.session_state.settings['use_web_search']
            )
            
            # Generate source highlights
            highlights = []
            if st.session_state.settings['show_source_highlights'] and self.document_viewer:
                highlights = self.document_viewer.highlight_sources(
                    query, retrieved_docs, crag_result.answer
                )
            
            # Evaluate response
            evaluation = {}
            if self.evaluation_system:
                evaluation = await self.evaluation_system.evaluate_answer(
                    query, crag_result.answer, retrieved_docs
                )
            
            # Refine knowledge
            if self.knowledge_system:
                await self.knowledge_system.refine_knowledge(query, retrieved_docs)
            
            return {
                'answer': crag_result.answer,
                'highlights': highlights,
                'metadata': {
                    'confidence': crag_result.confidence,
                    'method_used': crag_result.method_used,
                    'sources_count': len(crag_result.sources),
                    'evaluation': evaluation,
                    'processing_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in async query processing: {str(e)}")
            return {
                'answer': f"I apologize, but I encountered an error processing your question: {str(e)}",
                'highlights': [],
                'metadata': {'error': str(e)}
            }
    
    def render_right_panel(self):
        """Render the right panel with additional information."""
        if st.session_state.settings['show_source_highlights'] and st.session_state.source_highlights:
            st.subheader("🔍 Source Highlights")
            
            for i, highlight in enumerate(st.session_state.source_highlights[:5]):  # Show top 5
                with st.expander(f"{highlight.document_name} (Score: {highlight.relevance_score:.2f})"):
                    st.markdown(f"**Highlighted Text:** {highlight.highlighted_text}")
                    st.markdown(f"**Context:** ...{highlight.context_before}**{highlight.highlighted_text}**{highlight.context_after}...")
        
        # Document summary
        if st.session_state.documents:
            st.subheader("📚 Document Summary")
            st.write(f"Total documents: {len(st.session_state.documents)}")
            
            # Show recent documents
            for doc in st.session_state.documents[-3:]:  # Show last 3
                filename = doc.get('metadata', {}).get('filename', 'Unknown')
                content_length = doc.get('metadata', {}).get('content_length', 0)
                st.write(f"• {filename} ({content_length:,} chars)")
        
        # System status
        if st.session_state.settings['show_system_stats']:
            st.subheader("⚡ System Status")
            st.success("System Online")
            st.info(f"Documents: {len(st.session_state.documents)}")
            st.info(f"Chat History: {len(st.session_state.chat_history)}")


def main():
    """Main function to run the application."""
    try:
        app = MultiRAGChatbot()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
