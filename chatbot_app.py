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

# Import our RAG systems
from adaptive_rag import AdaptiveRAG
from crag import CRAG
from document_augmentation import DocumentProcessor, SentenceTransformerEmbeddings, load_document_content
from helper_functions import encode_document, replace_t_with_space
from explainable_retrieval import ExplainableRAGMethod

# Import evaluation framework
from evaluation_framework import EvaluationManager, UserFeedback
from analytics_dashboard import display_analytics_dashboard

# === PERSISTENT CHAT STORAGE ===
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

def load_rag_system(technique: str, document_paths: List[str] = None):
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
                    # CRAG works with single file, but we can use the climate change document
                    climate_doc = None
                    for doc_path in document_paths:
                        if "climate" in doc_path.lower() or "Understanding_Climate_Change" in doc_path:
                            climate_doc = doc_path
                            break
                    # If no climate doc found, use first document
                    selected_doc = climate_doc if climate_doc else document_paths[0]
                    return CRAG(selected_doc)
                else:
                    sample_file = "data/Understanding_Climate_Change (1).pdf"
                    if os.path.exists(sample_file):
                        return CRAG(sample_file)
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
    """Get response from the specified RAG system and return both response and context"""
    try:
        context = ""  # Will store retrieved context for evaluation
        
        if technique == "Adaptive RAG":
            # CRITICAL FIX: Use get_context_for_query() to get the EXACT context used for the answer
            try:
                # Get the exact context that will be used for generating the answer
                context = rag_system.get_context_for_query(query, silent=True)
                # Generate the answer using the same context
                response = rag_system.answer(query, silent=True)
                
                return response, context
            except Exception as e:
                # Fallback to old method if get_context_for_query is not available
                response = rag_system.answer(query)
                try:
                    # Try to extract context from the adaptive system if possible
                    docs = rag_system.get_relevant_documents(query)
                    if docs:
                        context = "\n".join([doc.page_content[:500] for doc in docs[:3]])
                    else:
                        context = "No specific document context retrieved"
                except:
                    # Fallback if get_relevant_documents is not available
                    context = "Context from uploaded documents"
                return response, context
        
        elif technique == "CRAG":
            try:
                st.info("🔄 Running CRAG analysis...")
                
                # Show what CRAG is doing
                st.write("**CRAG Process:**")
                st.write("1. Retrieving documents from your uploaded files...")
                st.write("2. Evaluating relevance to your query...")
                
                result = rag_system.run(query)
                st.success("✅ CRAG analysis completed")
                
                # Try to extract actual context from CRAG system
                try:
                    # Get the documents that CRAG likely used
                    docs = rag_system.get_relevant_documents(query)
                    if docs:
                        context = "\n".join([doc.page_content[:500] for doc in docs[:3]])
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
                
                return result, context
            except Exception as crag_error:
                error_msg = str(crag_error)
                st.error(f"❌ CRAG Error: {error_msg}")
                
                # Provide specific error guidance
                if "API" in error_msg or "google" in error_msg.lower():
                    st.warning("⚠️ This appears to be a Google API issue. Check your internet connection and API key.")
                elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    st.warning("⚠️ API rate limit reached. Please wait a moment and try again.")
                
                return f"CRAG failed with error: {error_msg}", ""
        
        elif technique == "Document Augmentation":
            # For document augmentation, we need to retrieve and generate answer
            docs = rag_system.get_relevant_documents(query)
            if docs:
                from document_augmentation import generate_answer
                context = docs[0].metadata.get('text', docs[0].page_content)
                response = generate_answer(context, query)
                return response, context
            else:
                return "No relevant documents found.", ""
        
        elif technique == "Basic RAG":
            # Basic similarity search
            docs = rag_system.similarity_search(query, k=3)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                # Simple context-based response
                response = f"Based on the documents:\n\n{context[:500]}..."
                return response, context
            else:
                return "No relevant documents found.", ""
        
        elif technique == "Explainable Retrieval":
            try:
                st.info("🔄 Running Explainable Retrieval...")
                
                # Show what Explainable Retrieval is doing
                st.write("**Explainable Retrieval Process:**")
                st.write("1. Retrieving relevant document chunks...")
                st.write("2. Generating explanations for each retrieved chunk...")
                st.write("3. Synthesizing a comprehensive answer with reasoning...")
                
                # Get detailed results for context
                detailed_results = rag_system.run(query)
                context = ""
                if detailed_results:
                    context = "\n".join([result['content'] for result in detailed_results])
                
                # Use the answer method for a comprehensive response
                answer = rag_system.answer(query)
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
                
                return answer, context
                    
            except Exception as er_error:
                error_msg = str(er_error)
                st.error(f"❌ Explainable Retrieval Error: {error_msg}")
                
                # Provide specific error guidance
                if "API" in error_msg or "google" in error_msg.lower():
                    st.warning("⚠️ This appears to be a Google API issue. Check your internet connection and API key.")
                elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    st.warning("⚠️ API rate limit reached. Please wait a moment and try again.")
                
                return f"Explainable Retrieval failed with error: {error_msg}", ""
                
    except Exception as e:
        return f"Error generating response: {str(e)}", ""

def add_message(role: str, content: str, technique: str = None, query_id: str = None):
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

def main():
    """Main application function"""
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
            
            st.session_state.uploaded_documents = document_paths
            st.success(f"Uploaded {len(document_paths)} document(s)")
              # Display uploaded files
            for i, file_path in enumerate(document_paths):
                st.write(f"📄 {os.path.basename(file_path)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
          # RAG Technique Selection
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
          # RAG Technique descriptions
        technique_descriptions = {
            "Adaptive RAG": "Dynamically adapts retrieval strategy based on query type (Factual, Analytical, Opinion, Contextual)",
            "CRAG": "Corrective RAG that evaluates retrieved documents and falls back to web search if needed",
            "Document Augmentation": "Enhances documents with generated questions for better retrieval",
            "Basic RAG": "Standard similarity-based retrieval and response generation",
            "Explainable Retrieval": "Provides explanations for why each retrieved document chunk is relevant to your query using Gemini AI"
        }
        
        st.markdown(f"""
        <div class="technique-card">            <strong>{selected_technique}</strong><br>
            <small>{technique_descriptions[selected_technique]}</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
          # Clear history button        # Session Management
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
            st.info("👋 Welcome! Upload some documents and ask me questions using different RAG techniques.")
        # Chat input
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            # Add user message
            add_message("user", query)
            
            # Load RAG system if not already loaded or documents changed
            if should_reload_rag_system(selected_technique, st.session_state.uploaded_documents):
                current_hash = get_document_hash(st.session_state.uploaded_documents)
                if current_hash != st.session_state.last_document_hash:
                    st.info("📄 Documents changed - reloading all RAG systems...")
                
                with st.spinner(f"Loading {selected_technique}..."):
                    rag_system = load_rag_system(selected_technique, st.session_state.uploaded_documents)
                    if rag_system:
                        st.session_state.rag_systems[selected_technique] = rag_system
                    else:
                        st.error(f"Failed to load {selected_technique}")
                        st.stop()
              # Get response with timing
            rag_system = st.session_state.rag_systems[selected_technique]
            start_time = time.time()
            
            with st.spinner(f"Generating response with {selected_technique}..."):
                response, context = get_rag_response(selected_technique, query, rag_system)
            
            processing_time = time.time() - start_time
            
            # Evaluate the response (Phase 2 & 3: Automated evaluation and storage)
            document_sources = [os.path.basename(doc) for doc in st.session_state.uploaded_documents]
            query_id = evaluation_manager.evaluate_rag_response(
                query=query,
                response=response,
                technique=selected_technique,
                document_sources=document_sources,
                context=context,  # Now passing the actual retrieved context
                processing_time=processing_time,
                session_id=st.session_state.session_id
            )
            
            # Add bot response with query_id for feedback linking
            add_message("assistant", response, selected_technique, query_id)
            
            # Mark this response as pending feedback (Phase 1: User feedback collection)
            st.session_state.pending_feedback[query_id] = True
            
            # Rerun to update the display
            st.rerun()
    
    with col2:
        st.header("📊 Statistics")
          # Message statistics
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        
        st.metric("Total Messages", total_messages)
        st.metric("Questions Asked", user_messages)
        st.metric("Documents Loaded", len(st.session_state.uploaded_documents))
        
        # Evaluation metrics preview
        st.subheader("📊 Performance Preview")
        comparison_data = evaluation_manager.get_technique_comparison()
        
        if comparison_data:
            # Show current session performance
            for technique, data in comparison_data.items():
                if data['total_queries'] > 0:
                    avg_rating = data.get('avg_user_rating', 0)
                    if avg_rating and not pd.isna(avg_rating):
                        st.metric(
                            f"{technique} Rating", 
                            f"{avg_rating:.1f}/5",
                            help=f"Average user rating based on {data.get('feedback_count', 0)} feedback(s)"
                        )
            
            st.info("💡 Visit the Analytics Dashboard for detailed performance insights!")
        else:
            st.info("📊 Performance metrics will appear here after you start chatting!")
        
        # Technique usage
        if st.session_state.messages:
            st.subheader("Technique Usage")
            technique_counts = {}
            for message in st.session_state.messages:
                if message["role"] == "assistant" and message["technique"]:
                    technique = message["technique"]
                    technique_counts[technique] = technique_counts.get(technique, 0) + 1
            
            for technique, count in technique_counts.items():
                st.write(f"• {technique}: {count}")
        
        # Export chat history
        if st.session_state.messages:
            st.subheader("💾 Export")
            if st.button("Download Chat History"):
                chat_data = {
                    "messages": st.session_state.messages,
                    "export_time": datetime.now().isoformat()
                }
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(chat_data, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
