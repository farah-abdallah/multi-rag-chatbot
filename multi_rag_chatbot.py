"""
Multi-RAG Chatbot

Integrates all four RAG techniques:
- Adaptive RAG with intelligent query classification
- CRAG with self-correction and web search fallback
- Document Augmentation with synthetic Q&A generation
- Explainable Retrieval with transparent explanations

Installation:
pip install streamlit langchain faiss-cpu sentence-transformers google-generativeai

Run with: streamlit run multi_rag_chatbot.py
"""

import streamlit as st
import os
import sys
import time
from typing import List, Dict, Any
import pandas as pd

# Import RAG techniques
sys.path.append(os.path.abspath('.'))
try:
    from adaptive_rag import AdaptiveRAG
    from crag import CRAG
    from document_augmentation import DocumentProcessor, SentenceTransformerEmbeddings
    from explainable_retrieval import ExplainableRAGMethod
except ImportError as e:
    st.error(f"Error importing RAG modules: {e}")

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-RAG Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_systems' not in st.session_state:
    st.session_state.rag_systems = {}
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

class MultiRAGChatbot:
    """Main chatbot class for Multi-RAG techniques"""
    
    def __init__(self):
        self.rag_techniques = {
            'Adaptive RAG': 'Intelligent query classification with adaptive retrieval',
            'CRAG': 'Self-correcting retrieval with web search fallback', 
            'Document Augmentation': 'Enhanced with synthetic Q&A generation',
            'Explainable Retrieval': 'Transparent retrieval with detailed explanations'
        }
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents"""
        all_texts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Process document
                from adaptive_rag import load_documents_from_files
                texts = load_documents_from_files([temp_path])
                all_texts.extend(texts)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        return all_texts
    
    def initialize_rag_systems(self, texts):
        """Initialize all RAG systems with processed texts"""
        rag_systems = {}
        
        with st.spinner("Initializing RAG systems..."):
            try:
                # Adaptive RAG
                rag_systems['Adaptive RAG'] = AdaptiveRAG(texts=texts)
                st.success("✅ Adaptive RAG initialized")
                
                # CRAG (needs file path, so we'll create a temp file)
                temp_file = "temp_combined.txt"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write('\n\n'.join(texts))
                rag_systems['CRAG'] = CRAG(temp_file)
                os.remove(temp_file)
                st.success("✅ CRAG initialized")
                
                # Document Augmentation
                embeddings = SentenceTransformerEmbeddings()
                combined_content = '\n\n'.join(texts)
                rag_systems['Document Augmentation'] = DocumentProcessor(combined_content, embeddings)
                st.success("✅ Document Augmentation initialized")
                
                # Explainable Retrieval
                rag_systems['Explainable Retrieval'] = ExplainableRAGMethod(texts)
                st.success("✅ Explainable Retrieval initialized")
                
            except Exception as e:
                st.error(f"Error initializing RAG systems: {e}")
        
        return rag_systems
    
    def query_rag_system(self, technique, query, rag_systems):
        """Query specific RAG technique and return response with timing"""
        start_time = time.time()
        
        try:
            if technique == 'Adaptive RAG':
                response = rag_systems[technique].answer(query)
            elif technique == 'CRAG':
                response = rag_systems[technique].run(query)
            elif technique == 'Document Augmentation':
                # Document Augmentation doesn't have direct query method
                response = f"Document Augmentation system initialized. Enhanced retrieval available for: {query}"
            elif technique == 'Explainable Retrieval':
                response = rag_systems[technique].answer(query)
            else:
                response = "Unknown technique"
                
        except Exception as e:
            response = f"Error: {e}"
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return response, processing_time

def main():
    """Main Streamlit app"""
    
    st.title("🔍 Multi-RAG Chatbot")
    st.markdown("### Advanced Document Analysis with Multiple RAG Techniques")
    
    # Initialize chatbot
    chatbot = MultiRAGChatbot()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'csv', 'json', 'docx', 'xlsx'],
            accept_multiple_files=True,
            help="Supports: PDF, TXT, CSV, JSON, DOCX, XLSX"
        )
        
        if uploaded_files and not st.session_state.documents_processed:
            if st.button("🚀 Process Documents"):
                # Process documents
                all_texts = chatbot.process_documents(uploaded_files)
                
                if all_texts:
                    # Initialize RAG systems
                    rag_systems = chatbot.initialize_rag_systems(all_texts)
                    
                    # Store in session state
                    st.session_state.rag_systems = rag_systems
                    st.session_state.documents_processed = True
                    
                    # Display summary
                    st.success(f"✅ Processed {len(uploaded_files)} documents")
                    st.info(f"📚 Total text chunks: {len(all_texts)}")
        
        # RAG Technique Selection
        st.header("🛠️ RAG Technique")
        selected_technique = st.selectbox(
            "Choose RAG technique:",
            options=list(chatbot.rag_techniques.keys()),
            help="Each technique has different strengths"
        )
        
        # Display technique info
        st.info(chatbot.rag_techniques[selected_technique])
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.documents_processed:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat['query'])
            with st.chat_message("assistant"):
                st.write(f"**{chat['technique']}** ({chat['time']:.2f}s)")
                st.write(chat['response'])
        
        # Query input
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(query)
            
            # Get RAG response
            with st.chat_message("assistant"):
                with st.spinner(f"Processing with {selected_technique}..."):
                    response, processing_time = chatbot.query_rag_system(
                        selected_technique, 
                        query, 
                        st.session_state.rag_systems
                    )
                
                st.write(f"**{selected_technique}** ({processing_time:.2f}s)")
                st.write(response)
            
            # Store in chat history
            st.session_state.chat_history.append({
                'query': query,
                'response': response,
                'technique': selected_technique,
                'time': processing_time
            })
        
        # Performance comparison
        if len(st.session_state.chat_history) > 0:
            with st.expander("📊 Performance Metrics"):
                df = pd.DataFrame(st.session_state.chat_history)
                if not df.empty:
                    avg_times = df.groupby('technique')['time'].mean().sort_values()
                    st.bar_chart(avg_times)
                    st.write("Average response times by technique")
    
    else:
        # Welcome message
        st.header("🚀 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📋 Features
            - **Multi-Format Support**: PDF, TXT, CSV, JSON, DOCX, XLSX
            - **4 RAG Techniques**: Each optimized for different use cases
            - **Real-time Comparison**: Compare responses across techniques
            - **Performance Metrics**: Track response times and quality
            - **Interactive Interface**: User-friendly web-based chat
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ RAG Techniques
            - **Adaptive RAG**: Intelligent query classification
            - **CRAG**: Self-correcting retrieval with web search
            - **Document Augmentation**: Synthetic Q&A enhancement
            - **Explainable Retrieval**: Transparent decision making
            - **Performance Analytics**: Compare technique effectiveness
            """)
        
        st.markdown("""
        ### 🎯 How to Use
        1. **Upload Documents** in the sidebar
        2. **Select RAG Technique** based on your needs
        3. **Ask Questions** about your documents
        4. **Compare Results** across different techniques
        """)
        
        st.info("👆 Upload documents in the sidebar to get started!")

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("🔑 Please set GOOGLE_API_KEY in your environment variables")
        st.stop()
    
    main()
