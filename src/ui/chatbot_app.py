"""
Multi-RAG Chatbot Application with Document Upload, Message History, and Comprehensive Evaluation

This application provides a web-based interface for testing different RAG techniques:
- Adaptive RAG
- CRAG (Corrective RAG)
- Document Augmentation RAG
- Basic RAG
- Explainable Retrieval RAG

Features:
- Upload documents (PDF, CSV, TXT, JSON, DOCX, XLSX)
- Choose RAG technique from dropdown
- Message history with technique tracking
- Comprehensive evaluation framework with user feedback and automated metrics
- Analytics dashboard for comparing technique performance
- Elegant, responsive UI
"""
import sys
import streamlit as st
import os
import tempfile
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import uuid
import sqlite3


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# === PERSISTENT CHAT STORAGE ===


# Import our RAG systems
from src.adaptive_rag import AdaptiveRAG
from src.crag import CRAG
from src.document_augmentation import DocumentProcessor, SentenceTransformerEmbeddings, load_document_content
from src.utils.helpers import encode_document, replace_t_with_space
from src.explainable_retrieval import ExplainableRAGMethod

# Import evaluation framework
from src.evaluation_framework import EvaluationManager, UserFeedback
from src.analytics_dashboard import display_analytics_dashboard

# Import document viewer components
from src.document_viewer import create_document_link, show_embedded_document_viewer, check_document_viewer_page

def init_chat_database():
    """Initialize database for persistent chat storage"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            message_id TEXT,
            message_type TEXT,
            content TEXT,
            technique TEXT,
            query_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_chat_message(session_id: str, message_id: str, message_type: str, content: str, technique: str = None, query_id: str = None):
    """Save chat message to database"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_sessions (session_id, message_id, message_type, content, technique, query_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, message_id, message_type, content, technique, query_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat message: {e}")

def load_chat_history(session_id: str):
    """Load chat history from database"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT message_id, message_type, content, technique, query_id, timestamp
            FROM chat_sessions 
            WHERE session_id = ?
            ORDER BY timestamp
        ''', (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            message_id, message_type, content, technique, query_id, timestamp = row
            message = {
                'id': message_id,
                'role': message_type,
                'content': content,
                'timestamp': timestamp
            }
            if technique:
                message['technique'] = technique
            if query_id:
                message['query_id'] = query_id
            messages.append(message)
        
        conn.close()
        return messages
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

def get_or_create_session_id():
    """Get existing session ID or create new one"""
    if 'persistent_session_id' not in st.session_state:
        st.session_state.persistent_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now()))}"
    return st.session_state.persistent_session_id

def auto_save_chat():
    """Auto-save current chat messages"""
    if 'messages' in st.session_state and st.session_state.messages:
        session_id = get_or_create_session_id()
        
        # Track what we've already saved
        if 'last_saved_count' not in st.session_state:
            st.session_state.last_saved_count = 0
        
        # Save only new messages
        new_messages = st.session_state.messages[st.session_state.last_saved_count:]
        
        for msg in new_messages:
            save_chat_message(
                session_id=session_id,
                message_id=msg.get('id', str(uuid.uuid4())),
                message_type=msg['role'],
                content=msg['content'],
                technique=msg.get('technique'),
                query_id=msg.get('query_id')
            )
        
        st.session_state.last_saved_count = len(st.session_state.messages)

def clear_current_session():
    """Clear current session chat history"""
    session_id = get_or_create_session_id()
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()
        
        # Clear session state
        st.session_state.messages = []
        st.session_state.last_saved_count = 0
        
    except Exception as e:
        print(f"Error clearing session: {e}")

def delete_chat_message(session_id: str, message_id: str):
    """Delete a specific chat message from database"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_sessions WHERE session_id = ? AND message_id = ?', (session_id, message_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error deleting chat message: {e}")

def delete_conversation_pair(user_message_id: str, assistant_message_id: str = None, query_id: str = None):
    """Delete a conversation pair (user question + assistant response) and associated ratings"""
    session_id = get_or_create_session_id()
    
    try:
        # Delete from chat history database
        delete_chat_message(session_id, user_message_id)
        if assistant_message_id:
            delete_chat_message(session_id, assistant_message_id)
        
        # Delete from evaluation database if query_id exists
        if query_id:
            evaluation_manager = get_evaluation_manager()
            evaluation_manager.delete_evaluation(query_id)
        
        # Remove from session state
        message_ids_to_delete = [user_message_id]
        if assistant_message_id:
            message_ids_to_delete.append(assistant_message_id)
            
        st.session_state.messages = [
            msg for msg in st.session_state.messages 
            if msg.get('id') not in message_ids_to_delete
        ]
        
        # Update last saved count
        st.session_state.last_saved_count = len(st.session_state.messages)
        
        return True
        
    except Exception as e:
        st.error(f"Error deleting conversation: {e}")
        return False

# === END PERSISTENT CHAT STORAGE ===

# Configure page
st.set_page_config(
    page_title="Multi-RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize evaluation manager
@st.cache_resource
def get_evaluation_manager():
    """Get or create evaluation manager"""
    return EvaluationManager()

# Custom CSS for elegant styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
      .technique-card {
        background: #f8f9fa;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .technique-card strong {
        color: #1a202c;
        font-weight: bold;
    }
    
    .technique-card small {
        color: #4a5568;
        line-height: 1.4;
    }
      .message-user {
        background: #e3f2fd;
        color: #1565c0;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
    }
    
    .message-user strong {
        color: #0d47a1;
        font-weight: bold;
    }
    
    .message-bot {
        background: #f3e5f5;
        color: #6a1b9a;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
        box-shadow: 0 2px 8px rgba(156, 39, 176, 0.1);
    }
    
    .message-bot strong {
        color: #4a148c;
        font-weight: bold;
    }
    
    .technique-badge {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
      .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .delete-button {
        background: #ff4757 !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 30px !important;
        height: 30px !important;
        font-size: 12px !important;
        cursor: pointer !important;
        margin-top: 0.5rem !important;
    }
    
    .delete-button:hover {
        background: #ff3742 !important;
        transform: scale(1.1) !important;
    }
    
    .source-reference {
        color: #0066cc !important;
        font-size: 0.85em !important;
        font-style: italic !important;
        background: rgba(0, 102, 204, 0.1) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border-left: 3px solid #0066cc !important;
        margin: 0 2px !important;
        display: inline-block !important;
    }
    
    .source-reference:hover {
        background: rgba(0, 102, 204, 0.2) !important;
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables with persistent chat history"""
    # Initialize chat database
    init_chat_database()
    
    # Get or create persistent session ID
    session_id = get_or_create_session_id()
    
    # Initialize messages with persistent storage
    if 'messages' not in st.session_state:
        # Try to load previous chat history
        saved_messages = load_chat_history(session_id)
        st.session_state.messages = saved_messages if saved_messages else []
        st.session_state.last_saved_count = len(st.session_state.messages)
    
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'rag_systems' not in st.session_state:
        st.session_state.rag_systems = {}
    if 'document_content' not in st.session_state:
        st.session_state.document_content = None
    if 'last_document_hash' not in st.session_state:
        st.session_state.last_document_hash = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'pending_feedback' not in st.session_state:
        st.session_state.pending_feedback = {}
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Chat"
    if 'last_source_chunks' not in st.session_state:
        st.session_state.last_source_chunks = {}  # Store source chunks by message ID
    
    # Update last activity timestamp
    st.session_state.last_activity = datetime.now()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return path"""
    try:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def load_rag_system(technique: str, document_paths: List[str] = None, crag_web_search_enabled: bool = None):
    """Load the specified RAG system with error handling"""
    try:
        with st.spinner(f"Loading {technique}..."):
            if technique == "Adaptive RAG":
                if document_paths:
                    return AdaptiveRAG(file_paths=document_paths)  # Pass all documents
                else:
                    return AdaptiveRAG(texts=["Sample text for testing. This is a basic RAG system."])
            
            elif technique == "CRAG":
                if document_paths and len(document_paths) > 0:
                    # Pass ALL uploaded document paths to CRAG for true multi-document retrieval
                    return CRAG(document_paths, web_search_enabled=crag_web_search_enabled)
                else:
                    sample_file = "data/Understanding_Climate_Change (1).pdf"
                    if os.path.exists(sample_file):
                        return CRAG([sample_file], web_search_enabled=crag_web_search_enabled)
                    else:
                        st.error("CRAG requires a document. Please upload a file first.")
                        return None
            
            elif technique == "Document Augmentation":
                if document_paths and len(document_paths) > 0:
                    # Process all documents and combine them
                    st.info(f"Processing {len(document_paths)} document(s) for Document Augmentation...")
                    
                    combined_content = ""
                    processed_docs = []
                    
                    for doc_path in document_paths:
                        try:
                            content = load_document_content(doc_path)
                            doc_name = os.path.basename(doc_path)
                            combined_content += f"\n\n=== Document: {doc_name} ===\n{content}"
                            processed_docs.append(doc_name)
                            st.success(f"✅ Processed: {doc_name}")
                        except Exception as e:
                            st.warning(f"⚠️ Skipped {os.path.basename(doc_path)}: {str(e)}")
                    
                    if combined_content:
                        st.info(f"Combined content from {len(processed_docs)} documents")
                        embedding_model = SentenceTransformerEmbeddings()
                        # Use the first document path for metadata, but process combined content
                        processor = DocumentProcessor(combined_content, embedding_model, document_paths[0])
                        return processor.run()
                    else:
                        st.error("No documents could be processed successfully.")
                        return None
                else:
                    st.error("Document Augmentation requires a document. Please upload a file first.")
                    return None
            
            elif technique == "Basic RAG":
                if document_paths and len(document_paths) > 0:
                    # For Basic RAG, we can use the new multi-document function
                    if len(document_paths) > 1:
                        st.info(f"Processing {len(document_paths)} documents with Basic RAG...")
                        # Use the new multi-document function
                        return create_multi_document_basic_rag(document_paths)
                    else:
                        return encode_document(document_paths[0])
                else:
                    st.error("Basic RAG requires a document. Please upload a file first.")
                    return None
            
            elif technique == "Explainable Retrieval":
                if document_paths and len(document_paths) > 0:
                    st.info(f"Processing {len(document_paths)} document(s) for Explainable Retrieval...")
                    
                    # Combine content from all documents
                    all_texts = []
                    processed_docs = []
                    
                    for doc_path in document_paths:
                        try:
                            content = load_document_content(doc_path)
                            doc_name = os.path.basename(doc_path)
                              # Split content into chunks for better retrieval
                            from langchain_text_splitters import RecursiveCharacterTextSplitter
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000, 
                                chunk_overlap=200
                            )
                            chunks = text_splitter.split_text(content)
                            
                            # Add source metadata to each chunk
                            for chunk in chunks:
                                all_texts.append(f"[Source: {doc_name}] {chunk}")
                            
                            processed_docs.append(doc_name)
                            st.success(f"✅ Processed: {doc_name}")
                        except Exception as e:
                            st.warning(f"⚠️ Skipped {os.path.basename(doc_path)}: {str(e)}")
                    
                    if all_texts:
                        st.info(f"Created explainable retrieval system with content from {len(processed_docs)} documents")
                        return ExplainableRAGMethod(all_texts)
                    else:
                        st.error("No documents could be processed successfully.")
                        return None
                else:
                    st.error("Explainable Retrieval requires a document. Please upload a file first.")
                    return None
                    
    except Exception as e:
        st.error(f"Error loading {technique}: {str(e)}")
        return None

def get_rag_response(technique: str, query: str, rag_system):
    """Get response from the specified RAG system and return response, context, and optionally source_chunks"""
    try:
        context = ""  # Will store retrieved context for evaluation
        source_chunks = None  # For CRAG responses
        
        if technique == "Adaptive RAG":
            # Build conversation context from previous messages
            conversation_context = ""
            if len(st.session_state.messages) > 1:  # If there's conversation history
                conversation_context = build_conversation_context(st.session_state.messages)
                st.write("Using conversation history for context-aware response...")
            
            # Create enhanced query with conversation context
            enhanced_query = query
            if conversation_context:
                enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
            
            # CRITICAL FIX: Use get_context_for_query() to get the EXACT context used for the answer
            try:
                # Get the exact context that will be used for generating the answer
                context = rag_system.get_context_for_query(enhanced_query, silent=True)
                # Generate the answer using the same context
                response = rag_system.answer(enhanced_query, silent=True)
                
                return response, context, None
            except Exception as e:
                # Fallback to old method if get_context_for_query is not available
                response = rag_system.answer(enhanced_query)
                try:
                    # Try to extract context from the adaptive system if possible
                    docs = rag_system.get_relevant_documents(enhanced_query)
                    if docs:
                        context = "\n".join([doc.page_content[:500] for doc in docs[:3]])
                    else:
                        context = "No specific document context retrieved"
                except:
                    # Fallback if get_relevant_documents is not available
                    context = "Context from uploaded documents"
                return response, context, None
        
        elif technique == "CRAG":
            try:
                st.info("🔄 Running CRAG analysis...")
                
                # Show what CRAG is doing
                st.write("**CRAG Process:**")
                st.write("1. Retrieving documents from your uploaded files...")
                st.write("2. Evaluating relevance to your query...")
                
                # Build conversation context from previous messages
                conversation_context = ""
                if len(st.session_state.messages) > 1:  # If there's conversation history
                    conversation_context = build_conversation_context(st.session_state.messages)
                    st.write("3. Considering conversation history for context-aware response...")
                
                # Create enhanced query with conversation context
                enhanced_query = query
                if conversation_context:
                    enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
                
                # Use the enhanced run method that returns source chunks
                if hasattr(rag_system, 'run_with_sources'):
                    result_data = rag_system.run_with_sources(enhanced_query)
                    response = result_data['answer']
                    source_chunks = result_data['source_chunks']
                    sources = result_data['sources']
                    
                    st.success("✅ CRAG analysis completed")
                    
                    # Extract context for evaluation
                    context = "\n".join([chunk['text'] for chunk in source_chunks])
                    
                    # Return response and context, source chunks will be stored via add_message
                    return response, context, source_chunks
                    
                else:
                    # Fallback to original method
                    result = rag_system.run(enhanced_query)
                    response = result
                    st.success("✅ CRAG analysis completed")
                    
                    # Try to extract context from CRAG system
                    try:
                        # Get the documents that CRAG likely used
                        docs = rag_system.vectorstore.similarity_search(query, k=3)
                        if docs:
                            context = "\n".join([doc.page_content[:500] for doc in docs])
                        else:
                            context = "CRAG used web search or external sources"
                    except:
                        # Fallback - extract from result if it contains source information
                        context = str(result)[:1000] if result else "CRAG context not available"
                
                # Add explanation of what CRAG did
                st.info("""
                **How CRAG worked for this query:**
                - If relevance was HIGH (>0.7): Used your uploaded document
                - If relevance was LOW (<0.3): Performed web search instead  
                - If relevance was MEDIUM (0.3-0.7): Combined both sources
                
                Check the response to see which source(s) were actually used!
                """)
                
                return response, context, None
            except Exception as crag_error:
                error_msg = str(crag_error)
                st.error(f"❌ CRAG Error: {error_msg}")
                
                # Provide specific error guidance
                if "API" in error_msg or "google" in error_msg.lower():
                    st.warning("⚠️ This appears to be a Google API issue. Check your internet connection and API key.")
                elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    st.warning("⚠️ API rate limit reached. Please wait a moment and try again.")
                
                return f"CRAG failed with error: {error_msg}", "", None
        
        elif technique == "Document Augmentation":
            # Build conversation context from previous messages
            conversation_context = ""
            if len(st.session_state.messages) > 1:  # If there's conversation history
                conversation_context = build_conversation_context(st.session_state.messages)
                st.write("Using conversation history for context-aware response...")
            
            # Create enhanced query with conversation context
            enhanced_query = query
            if conversation_context:
                enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
            
            # For document augmentation, we need to retrieve and generate answer
            docs = rag_system.get_relevant_documents(enhanced_query)
            if docs:
                from document_augmentation import generate_answer
                context = docs[0].metadata.get('text', docs[0].page_content)
                response = generate_answer(context, enhanced_query)
                return response, context, None
            else:
                return "No relevant documents found.", "", None
        
        elif technique == "Basic RAG":
            # Build conversation context from previous messages
            conversation_context = ""
            if len(st.session_state.messages) > 1:  # If there's conversation history
                conversation_context = build_conversation_context(st.session_state.messages)
                st.write("Using conversation history for context-aware response...")
            
            # Create enhanced query with conversation context
            enhanced_query = query
            if conversation_context:
                enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
            
            # Basic similarity search
            docs = rag_system.similarity_search(enhanced_query, k=3)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                # Simple context-based response
                response = f"Based on the documents:\n\n{context[:500]}..."
                return response, context, None
            else:
                return "No relevant documents found.", "", None
        
        elif technique == "Explainable Retrieval":
            try:
                st.info("🔄 Running Explainable Retrieval...")
                
                # Show what Explainable Retrieval is doing
                st.write("**Explainable Retrieval Process:**")
                st.write("1. Retrieving relevant document chunks...")
                st.write("2. Generating explanations for each retrieved chunk...")
                st.write("3. Synthesizing a comprehensive answer with reasoning...")
                
                # Build conversation context from previous messages
                conversation_context = ""
                if len(st.session_state.messages) > 1:  # If there's conversation history
                    conversation_context = build_conversation_context(st.session_state.messages)
                    st.write("4. Considering conversation history for context-aware response...")
                
                # Create enhanced query with conversation context
                enhanced_query = query
                if conversation_context:
                    enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
                
                # Get detailed results for context
                detailed_results = rag_system.run(enhanced_query)
                context = ""
                if detailed_results:
                    context = "\n".join([result['content'] for result in detailed_results])
                
                # Use the answer method for a comprehensive response
                answer = rag_system.answer(enhanced_query)
                st.success("✅ Explainable Retrieval completed")
                
                # Also show the detailed explanations in an expander
                with st.expander("🔍 View Detailed Explanations"):
                    if detailed_results:
                        for i, result in enumerate(detailed_results, 1):
                            st.write(f"**📄 Retrieved Section {i}:**")
                            st.write(f"**Content:** {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")
                            st.write(f"**💡 Explanation:** {result['explanation']}")
                            st.write("---")
                    else:
                        st.write("No detailed explanations available.")
                
                return answer, context, None
                    
            except Exception as er_error:
                error_msg = str(er_error)
                st.error(f"❌ Explainable Retrieval Error: {error_msg}")
                
                # Provide specific error guidance
                if "API" in error_msg or "google" in error_msg.lower():
                    st.warning("⚠️ This appears to be a Google API issue. Check your internet connection and API key.")
                elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    st.warning("⚠️ API rate limit reached. Please wait a moment and try again.")
                
                return f"Explainable Retrieval failed with error: {error_msg}", "", None
                
    except Exception as e:
        return f"Error generating response: {str(e)}", "", None

def add_message(role: str, content: str, technique: str = None, query_id: str = None, source_chunks: list = None):
    """Add message to session state and save to database"""
    message = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "technique": technique,
        "query_id": query_id,  # For linking with evaluation
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(message)
    
    # Store source chunks if provided (for CRAG responses)
    if source_chunks and role == "assistant":
        st.session_state.last_source_chunks[message["id"]] = source_chunks
    
    # Auto-save to persistent storage
    auto_save_chat()

def collect_user_feedback(query_id: str, message_id: str):
    """Collect user feedback for a specific response"""
    with st.expander("📝 Rate this response", expanded=False):
        st.write("Help us improve by rating this response:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            helpfulness = st.slider(
                "How helpful was this response?",
                min_value=1, max_value=5, value=3,
                key=f"helpfulness_{message_id}"
            )
            
            accuracy = st.slider(
                "How accurate was this response?",
                min_value=1, max_value=5, value=3,
                key=f"accuracy_{message_id}"
            )
        
        with col2:
            clarity = st.slider(
                "How clear was this response?",
                min_value=1, max_value=5, value=3,
                key=f"clarity_{message_id}"
            )
            
            overall_rating = st.slider(
                "Overall rating",
                min_value=1, max_value=5, value=3,
                key=f"overall_{message_id}"
            )
        
        comments = st.text_area(
            "Additional comments (optional):",
            key=f"comments_{message_id}",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Submit Feedback", key=f"submit_{message_id}"):
                # Create feedback object
                feedback = UserFeedback(
                    helpfulness=helpfulness,
                    accuracy=accuracy,
                    clarity=clarity,
                    overall_rating=overall_rating,
                    comments=comments,
                    timestamp=datetime.now().isoformat()
                )
                
                # Submit feedback to evaluation manager
                evaluation_manager = get_evaluation_manager()
                evaluation_manager.add_user_feedback(query_id, feedback)
                
                st.success("Thank you for your feedback! 🙏")
                
                # Remove from pending feedback
                if query_id in st.session_state.pending_feedback:
                    del st.session_state.pending_feedback[query_id]
                
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("Skip", key=f"skip_{message_id}"):
                # Remove from pending feedback without submitting
                if query_id in st.session_state.pending_feedback:
                    del st.session_state.pending_feedback[query_id]
                st.rerun()

def display_source_documents(message_id: str, source_chunks: List[Dict]):
    """Display source document links for a specific message"""
    if not source_chunks:
        return
    
    # Group chunks by document
    docs_with_chunks = {}
    for chunk in source_chunks:
        doc_path = chunk['source']
        if doc_path not in docs_with_chunks:
            docs_with_chunks[doc_path] = []
        docs_with_chunks[doc_path].append(chunk)
    
    # Create links for each document
    st.markdown("### 📄 Source Documents:")
    for doc_path, chunks in docs_with_chunks.items():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            doc_name = os.path.basename(doc_path) if doc_path != 'Unknown' else 'Uploaded Document'
            st.write(f"**{doc_name}** - {len(chunks)} chunk(s) used")
            avg_score = sum(chunk.get('score', 0) for chunk in chunks) / len(chunks)
            st.caption(f"Average relevance score: {avg_score:.2f}")
        with col2:
            # Button to view in new tab
            if doc_path != 'Unknown':
                create_document_link(
                    doc_path, 
                    chunks, 
                    "🔗 New Tab"
                )
        with col3:
            # Button to view embedded - use message_id to make keys unique
            embed_key = f"embed_{message_id}_{hash(doc_path)}"
            show_key = f'show_doc_{message_id}_{hash(doc_path)}'
            if st.button(f"👁️ View Here", key=embed_key):
                st.session_state[show_key] = True
                st.rerun()
    
    # Show embedded viewers if requested
    for doc_path, chunks in docs_with_chunks.items():
        show_key = f'show_doc_{message_id}_{hash(doc_path)}'
        if st.session_state.get(show_key, False):
            show_embedded_document_viewer(doc_path, chunks, use_expander=False, message_id=message_id)
            hide_key = f"hide_{message_id}_{hash(doc_path)}"
            if st.button(f"❌ Hide {os.path.basename(doc_path)}", key=hide_key):
                st.session_state[show_key] = False
                st.rerun()

def display_message(message: Dict[str, Any], message_index: int = None):
    """Display a single message with delete functionality"""
    if message["role"] == "user":
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.markdown(f"""
            <div class="message-user">
                <strong>You:</strong> {message["content"]}
                <br><small>{datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Add delete button for user messages
            if st.button("🗑️", key=f"delete_user_{message['id']}", help="Delete this question and response"):
                # Find the corresponding assistant message
                assistant_message = None
                if message_index is not None and message_index + 1 < len(st.session_state.messages):
                    next_message = st.session_state.messages[message_index + 1]
                    if next_message["role"] == "assistant":
                        assistant_message = next_message
                
                if assistant_message:
                    # Delete the conversation pair
                    query_id = assistant_message.get("query_id")
                    success = delete_conversation_pair(
                        message["id"], 
                        assistant_message["id"], 
                        query_id
                    )
                    if success:
                        st.success("Question and response deleted!")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    # Delete just the user message if no assistant response
                    success = delete_conversation_pair(message["id"], None, None)
                    if success:
                        st.success("Question deleted!")
                        time.sleep(0.5)
                        st.rerun()
    else:
        technique_badge = f'<span class="technique-badge">{message["technique"]}</span>' if message["technique"] else ''
        st.markdown(f"""
        <div class="message-bot">
            <strong>Assistant:</strong> {technique_badge}
            <br>{message["content"]}
            <br><small>{datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Show feedback collection for bot messages that have a query_id and are pending feedback
        query_id = message.get("query_id")
        if query_id and query_id in st.session_state.pending_feedback:
            collect_user_feedback(query_id, message["id"])
        
        # Show source documents for CRAG responses
        if message.get("technique") == "CRAG" and message["id"] in st.session_state.last_source_chunks:
            source_chunks = st.session_state.last_source_chunks[message["id"]]
            if source_chunks:
                display_source_documents(message["id"], source_chunks)

def create_multi_document_basic_rag(document_paths: List[str], chunk_size=1000, chunk_overlap=200):
    """
    Create a Basic RAG system that can handle multiple documents by combining them into a single vectorstore.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    
    try:
        all_documents = []
        processed_files = []
        
        # Load and process each document
        for file_path in document_paths:
            try:
                # Load content from the document
                content = load_document_content(file_path)
                doc_name = os.path.basename(file_path)
                
                # Create a document object with metadata
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path, "filename": doc_name}
                )
                all_documents.append(doc)
                processed_files.append(doc_name)
                
            except Exception as e:
                st.warning(f"⚠️ Could not process {os.path.basename(file_path)}: {str(e)}")
                continue
        
        if not all_documents:
            raise ValueError("No documents could be processed successfully")
        
        st.success(f"✅ Loaded {len(processed_files)} documents: {', '.join(processed_files)}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=len
        )
        texts = text_splitter.split_documents(all_documents)
        
        # Clean the texts (remove tab characters)
        cleaned_texts = replace_t_with_space(texts)
        
        # Create embeddings and vector store using local embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
        
        st.info(f"Created vectorstore with {len(cleaned_texts)} chunks from {len(processed_files)} documents")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating multi-document Basic RAG: {str(e)}")
        return None

def get_document_hash(document_paths):
    """Create a hash of the current document list to detect changes"""
    if not document_paths:
        return None
    # Sort paths and create a simple hash
    sorted_paths = sorted(document_paths)
    return hash(tuple(sorted_paths))

def should_reload_rag_system(technique, document_paths):
    """Check if RAG system should be reloaded due to document changes"""
    current_hash = get_document_hash(document_paths)
    
    # If documents changed, clear all cached systems
    if current_hash != st.session_state.last_document_hash:
        st.session_state.rag_systems = {}
        st.session_state.last_document_hash = current_hash
        return True
    
    # If system not loaded for this technique, need to load
    return technique not in st.session_state.rag_systems

def build_conversation_context(messages, max_turns=3):
    """
    Build a context string from the last N conversation turns.
    This allows the chatbot to remember and reference previous exchanges.
    
    Args:
        messages: List of message dictionaries from session state
        max_turns: Maximum number of conversation turns to include
        
    Returns:
        String with formatted conversation history
    """
    # Only use recent messages (limited by max_turns)
    recent_messages = messages[-max_turns*2:] if len(messages) > max_turns*2 else messages
    
    # Format the conversation
    conversation_lines = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_lines.append(f"{role}: {msg['content']}")
    # Join into a single string with line breaks
    conversation_context = "\n".join(conversation_lines)
    return conversation_context

def create_multi_document_basic_rag(document_paths: List[str], chunk_size=1000, chunk_overlap=200):
    """
    Create a Basic RAG system that can handle multiple documents by combining them into a single vectorstore.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    
    try:
        all_documents = []
        processed_files = []
        
        # Load and process each document
        for file_path in document_paths:
            try:
                content = load_document_content(file_path)
                file_name = os.path.basename(file_path)
                
                # Create a document with metadata
                doc = Document(
                    page_content=content,
                    metadata={"source": file_name, "file_path": file_path}
                )
                all_documents.append(doc)
                processed_files.append(file_name)
                
            except Exception as e:
                st.warning(f"⚠️ Could not process {os.path.basename(file_path)}: {str(e)}")
        
        if not all_documents:
            raise ValueError("No documents could be processed successfully")
        
        st.success(f"✅ Loaded {len(processed_files)} documents: {', '.join(processed_files)}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=len
        )
        texts = text_splitter.split_documents(all_documents)
        
        # Clean the texts (remove tab characters)
        cleaned_texts = replace_t_with_space(texts)
        
        # Create embeddings and vector store using local embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
        
        st.info(f"Created vectorstore with {len(cleaned_texts)} chunks from {len(processed_files)} documents")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating multi-document Basic RAG: {str(e)}")
        return None

def get_document_hash(document_paths):
    """Create a hash of the current document list to detect changes"""
    if not document_paths:
        return None
    # Sort paths and create a simple hash
    sorted_paths = sorted(document_paths)
    return hash(tuple(sorted_paths))

def should_reload_rag_system(technique, document_paths):
    """Check if RAG system should be reloaded due to document changes"""
    current_hash = get_document_hash(document_paths)
    
    # If documents changed, clear all cached systems
    if current_hash != st.session_state.last_document_hash:
        st.session_state.rag_systems = {}
        st.session_state.last_document_hash = current_hash
        return True
    
    # If system not loaded for this technique, need to load
    return technique not in st.session_state.rag_systems

def build_conversation_context(messages, max_turns=3):
    """
    Build a context string from the last N conversation turns.
    This allows the chatbot to remember and reference previous exchanges.
    
    Args:
        messages: List of message dictionaries from session state
        max_turns: Maximum number of conversation turns to include
        
    Returns:
        String with formatted conversation history
    """
    # Only use recent messages (limited by max_turns)
    recent_messages = messages[-max_turns*2:] if len(messages) > max_turns*2 else messages
    
    # Format the conversation
    conversation_lines = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_lines.append(f"{role}: {msg['content']}")
    
    # Join into a single string with line breaks
    conversation_context = "\n".join(conversation_lines)
    return conversation_context
# === CHAT SESSION MANAGEMENT ===

def get_all_chat_sessions():
    """Get all chat sessions with their first message as title"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, MIN(timestamp) as first_message_time,
                   (SELECT content FROM chat_sessions cs2 
                    WHERE cs2.session_id = cs1.session_id 
                    AND cs2.message_type = 'user' 
                    ORDER BY timestamp ASC LIMIT 1) as first_message
            FROM chat_sessions cs1
            GROUP BY session_id
            ORDER BY first_message_time DESC
        ''')
        
        sessions = []
        for row in cursor.fetchall():
            session_id, timestamp, first_message = row
            # Create a readable title from the first message
            if first_message:
                title = first_message[:50] + "..." if len(first_message) > 50 else first_message
            else:
                title = f"Chat {session_id[-8:]}"
            
            sessions.append({
                'session_id': session_id,
                'title': title,
                'timestamp': timestamp,
                'first_message': first_message
            })
        
        conn.close()
        return sessions
    except Exception as e:
        print(f"Error getting chat sessions: {e}")
        return []

def create_new_chat_session():
    """Create a new chat session"""
    new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now()))}"
    st.session_state.persistent_session_id = new_session_id
    st.session_state.messages = []
    st.session_state.last_saved_count = 0
    st.session_state.pending_feedback = {}
    return new_session_id

def switch_to_chat_session(session_id):
    """Switch to a specific chat session"""
    if session_id != st.session_state.get('persistent_session_id'):
        st.session_state.persistent_session_id = session_id
        st.session_state.messages = load_chat_history(session_id)
        st.session_state.last_saved_count = len(st.session_state.messages)
        st.session_state.pending_feedback = {}
        return True
    return False

def delete_chat_session(session_id):
    """Delete an entire chat session"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()
        
        # If we're deleting the current session, create a new one
        if session_id == st.session_state.get('persistent_session_id'):
            create_new_chat_session()
        
        return True
    except Exception as e:
        print(f"Error deleting chat session: {e}")
        return False

def rename_chat_session(session_id, new_title):
    """Rename a chat session by updating its first message"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        # Update the first user message content to serve as the title
        cursor.execute('''
            UPDATE chat_sessions 
            SET content = ?
            WHERE session_id = ?
            AND message_type = 'user'
            AND timestamp = (
                SELECT MIN(timestamp) 
                FROM chat_sessions 
                WHERE session_id = ? AND message_type = 'user'
            )
        ''', (new_title, session_id, session_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error renaming chat session: {e}")
        return False

# === END CHAT SESSION MANAGEMENT ===

def main():
    """Main application function"""
    # Check if this is a document viewer page first
    if check_document_viewer_page():
        return
        
    initialize_session_state()
    
    # Get evaluation manager
    evaluation_manager = get_evaluation_manager()
    
    # Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Choose page:",
        ["💬 Chat", "📊 Analytics Dashboard"],
        index=0 if st.session_state.current_page == "Chat" else 1
    )
    
    # Update current page
    if page == "💬 Chat":
        st.session_state.current_page = "Chat"
    else:
        st.session_state.current_page = "Analytics"
    
    if st.session_state.current_page == "Analytics":
        # Show analytics dashboard
        display_analytics_dashboard(evaluation_manager)
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Multi-RAG Chatbot with Evaluation</h1>
        <p>Compare different RAG techniques with your documents and get comprehensive analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for document upload and RAG selection
    with st.sidebar:
        # === CHAT SESSION SELECTOR ===
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("💬 Chat Sessions")
        
        # Get all chat sessions
        all_sessions = get_all_chat_sessions()
        current_session_id = get_or_create_session_id()
        
        # Create new chat button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("➕ New Chat", use_container_width=True):
                create_new_chat_session()
                st.rerun()
        
        with col2:
            if st.button("🔄", help="Refresh chat list"):
                st.rerun()
        
        # Display chat sessions
        if all_sessions:
            st.write("**Select a chat:**")
            for i, session in enumerate(all_sessions):
                is_current = session['session_id'] == current_session_id
                
                # Create a container for each chat session
                with st.container():
                    col1, col2, col3 = st.columns([6, 1, 1])
                    
                    with col1:
                        # Chat selection button
                        button_style = "🔵" if is_current else "⚪"
                        chat_label = f"{button_style} {session['title']}"
                        
                        if st.button(
                            chat_label,
                            key=f"chat_{session['session_id']}",
                            help=f"Switch to this chat\nCreated: {session['timestamp'][:19]}",
                            use_container_width=True
                        ):
                            if switch_to_chat_session(session['session_id']):
                                st.rerun()
                    
                    with col2:
                        # Rename button
                        if st.button("✏️", key=f"rename_{session['session_id']}", help="Rename chat"):
                            st.session_state[f"rename_mode_{session['session_id']}"] = True
                            st.rerun()
                    
                    with col3:
                        # Delete button
                        if st.button("🗑️", key=f"delete_{session['session_id']}", help="Delete chat"):
                            if st.session_state.get(f"confirm_delete_{session['session_id']}", False):
                                if delete_chat_session(session['session_id']):
                                    st.success("Chat deleted!")
                                    st.session_state[f"confirm_delete_{session['session_id']}"] = False
                                    st.rerun()
                            else:
                                st.session_state[f"confirm_delete_{session['session_id']}"] = True
                                st.warning("Click again to confirm deletion")
                    
                    # Rename input field
                    if st.session_state.get(f"rename_mode_{session['session_id']}", False):
                        new_title = st.text_input(
                            "New chat title:",
                            value=session['title'],
                            key=f"rename_input_{session['session_id']}"
                        )
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("Save", key=f"save_rename_{session['session_id']}"):
                                if rename_chat_session(session['session_id'], new_title):
                                    st.success("Chat renamed!")
                                    st.session_state[f"rename_mode_{session['session_id']}"] = False
                                    st.rerun()
                        
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_rename_{session['session_id']}"):
                                st.session_state[f"rename_mode_{session['session_id']}"] = False
                                st.rerun()
                    
                    # Show confirmation message for delete
                    if st.session_state.get(f"confirm_delete_{session['session_id']}", False):
                        st.warning("⚠️ Are you sure? This will permanently delete this chat.")
                
                # Add a separator line
                if i < len(all_sessions) - 1:
                    st.markdown("---")
        else:
            st.info("No chat sessions yet. Start a new chat!")
        
        # Current session info
        if st.session_state.messages:
            st.markdown("**Current Chat:**")
            st.caption(f"💬 {len(st.session_state.messages)} messages")
            st.caption(f"🔑 {current_session_id[-8:]}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        # === END CHAT SESSION SELECTOR ===
        
        # === DOCUMENT UPLOAD SECTION ===
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'csv', 'json', 'docx', 'xlsx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, CSV, JSON, DOCX, XLSX"
        )
        
        if uploaded_files:
            # Save uploaded files
            document_paths = []
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    document_paths.append(file_path)
            
            if document_paths:
                st.session_state.uploaded_documents = document_paths
                st.success(f"Uploaded {len(document_paths)} document(s)")
                
                # Display uploaded files
                for i, file_path in enumerate(document_paths):
                    st.write(f"📄 {os.path.basename(file_path)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === RAG TECHNIQUE SELECTION ===
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔧 RAG Technique")
        
        rag_techniques = [
            "Adaptive RAG",
            "CRAG",
            "Document Augmentation",
            "Basic RAG",
            "Explainable Retrieval"
        ]
        
        selected_technique = st.selectbox(
            "Choose RAG technique:",
            rag_techniques,
            help="Select the RAG technique to use for answering questions"
        )
        
        # Show CRAG web search toggle only if CRAG is selected
        if selected_technique == "CRAG":
            crag_web_search_enabled = st.checkbox(
                "Enable web search fallback (CRAG)",
                value=True,
                help="If enabled, CRAG will use web search when your document is insufficient. Disable to use only your uploaded document."
            )
        else:
            crag_web_search_enabled = None
        
        # RAG Technique descriptions
        technique_descriptions = {
            "Adaptive RAG": "Dynamically adapts retrieval strategy based on query type (Factual, Analytical, Opinion, Contextual)",
            "CRAG": "Corrective RAG that evaluates retrieved documents and falls back to web search if needed",
            "Document Augmentation": "Enhances documents with generated questions for better retrieval",
            "Basic RAG": "Standard similarity-based retrieval and response generation",
            "Explainable Retrieval": "Provides explanations for why each retrieved document chunk is relevant to your query using Gemini AI"
        }
        
        st.markdown(f"""
        <div class="technique-card">
            <strong>{selected_technique}</strong><br>
            <small>{technique_descriptions[selected_technique]}</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === SESSION MANAGEMENT ===
        st.markdown("### 💾 Session Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                clear_current_session()
                st.success("✅ Chat cleared and saved!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Recover Last"):
                # Load the most recent session
                try:
                    conn = sqlite3.connect('chat_history.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT DISTINCT session_id FROM chat_sessions 
                        ORDER BY timestamp DESC LIMIT 1
                    ''')
                    
                    result = cursor.fetchone()
                    if result and result[0] != get_or_create_session_id():
                        latest_session = result[0]
                        recovered_messages = load_chat_history(latest_session)
                        
                        if recovered_messages:
                            st.session_state.messages = recovered_messages
                            st.session_state.last_saved_count = len(recovered_messages)
                            st.success(f"🔄 Recovered {len(recovered_messages)} messages!")
                            st.rerun()
                        else:
                            st.warning("No messages to recover")
                    else:
                        st.warning("No previous sessions found")
                    
                    conn.close()
                except Exception as e:
                    st.error(f"Error recovering session: {e}")
        
        # Conversation Management Help
        if st.session_state.messages:
            st.markdown("### 🗂️ Individual Message Management")
            st.info("""
            **💡 Tip:** Click the 🗑️ button next to any question to delete that specific question and its response, including any ratings you gave.
            
            This is useful for:
            - Removing test questions
            - Cleaning up incorrect queries
            - Managing chat history length
            """)
            
            # Show current stats
            user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
            assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
            
            st.markdown(f"""
            **Current Session:**
            - 💬 {len(user_messages)} questions asked
            - 🤖 {len(assistant_messages)} responses given
            - 📊 {len([m for m in assistant_messages if m.get('query_id')])} responses available for rating
            """)
        
        # Show session info
        if st.session_state.messages:
            session_id = get_or_create_session_id()
            st.caption(f"💬 {len(st.session_state.messages)} messages")
            st.caption(f"🔑 Session: {session_id[-8:]}")
        
        st.markdown("### 📊 Analytics")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Clear Analytics"):
                # Import the clear function
                from analytics_dashboard import perform_database_reset
                
                if st.session_state.get('confirm_clear_analytics_sidebar', False):
                    perform_database_reset(evaluation_manager)
                    st.session_state.confirm_clear_analytics_sidebar = False
                else:
                    st.session_state.confirm_clear_analytics_sidebar = True
                    st.warning("⚠️ Click again to confirm clearing ALL analytics data")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Header with session status
        col_header, col_status = st.columns([3, 1])
        with col_header:
            st.header("💬 Chat")
        with col_status:
            if st.session_state.messages:
                st.caption(f"💾 Auto-saved ({len(st.session_state.messages)} msgs)")
            else:
                st.caption("💾 Session ready")
        
        # Display messages
        if st.session_state.messages:
            for index, message in enumerate(st.session_state.messages):
                display_message(message, index)
        else:
            st.info("👋 Welcome! Upload documents and start asking questions using different RAG techniques.")
        
        # Chat input
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            # Prevent stale highlighting
            for key in list(st.session_state.keys()):
                if key.startswith('highlight_'):
                    del st.session_state[key]
            
            # Add user message
            add_message("user", query)
            
            # Check if RAG system already loaded or documents changed
            if get_document_hash(st.session_state.uploaded_documents) != st.session_state.last_document_hash:
                st.info("📄 Documents changed - reloading all RAG systems...")
                
            # Load RAG system
            with st.spinner(f"Loading {selected_technique}..."):
                rag_system = load_rag_system(selected_technique, st.session_state.uploaded_documents, crag_web_search_enabled)
            
            if rag_system:
                # Store RAG system for future use
                st.session_state.rag_systems[selected_technique] = rag_system
                
                # Generate response timing
                start_time = time.time()
                
                with st.spinner(f"Generating response with {selected_technique}..."):
                    response, context, source_chunks = get_rag_response(selected_technique, query, rag_system)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Add assistant message (we'll get the query_id from evaluation)
                add_message("assistant", response, selected_technique, None, source_chunks)
                
                # Store evaluation data (automatic evaluation)
                try:
                    document_sources = [os.path.basename(doc) for doc in st.session_state.uploaded_documents]
                    query_id = evaluation_manager.evaluate_rag_response(
                        query=query,
                        response=response,
                        context=context,  # The retrieved context
                        technique=selected_technique,
                        processing_time=response_time,
                        document_sources=document_sources,
                        session_id=get_or_create_session_id()
                    )
                    
                    # Update the assistant message with the query_id
                    if st.session_state.messages:
                        st.session_state.messages[-1]["query_id"] = query_id
                    
                    # Mark this response as pending feedback (Phase 1: User feedback collection)
                    st.session_state.pending_feedback[query_id] = True
                    
                    st.rerun()
                except Exception as eval_error:
                    st.warning(f"Response generated but evaluation storage failed: {eval_error}")
        
        # Show basic statistics
        with col2:
            st.subheader("📊 Statistics")
            
            # Message statistics
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            total_messages = len(st.session_state.messages)
            
            st.metric("Questions", user_messages)
            st.metric("Responses", assistant_messages)
            st.metric("Documents", len(st.session_state.uploaded_documents))
            
            # Evaluation metrics preview
            if st.session_state.messages:
                comparison_data = evaluation_manager.get_technique_comparison()
                
                for technique, data in comparison_data.items():
                    if data['total_queries'] > 0:
                        avg_rating = data.get('avg_user_rating', 0)
                        if avg_rating and not pd.isna(avg_rating):
                            st.metric(f"{technique} Rating", 
                                    f"{avg_rating:.1f}/5", 
                                    help=f"Average user rating based on {data.get('feedback_count', 0)} feedback(s)"
                            )
            
            st.info("💡 Visit the Analytics Dashboard for detailed performance insights!")
            
            if not st.session_state.messages:
                st.info("📝 Your conversation statistics will appear here after you start chatting!")
        
        # Export functionality
        if st.session_state.messages:
            st.subheader("Technique Usage")
            technique_counts = {}
            for message in st.session_state.messages:
                if message["role"] == "assistant" and message["technique"]:
                    technique = message["technique"]
                    technique_counts[technique] = technique_counts.get(technique, 0) + 1
            
            for technique, count in technique_counts.items():
                st.write(f"**{technique}**: {count}")
            
            st.subheader("💾 Export")
            if st.button("📄 Download Chat History"):
                export_data = {
                    "messages": st.session_state.messages,
                    "session_id": get_or_create_session_id(),
                    "export_time": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()