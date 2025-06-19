"""
Advanced Multi-RAG Chatbot with Comparison Features

Enhanced version of the chatbot that allows side-by-side comparison of different RAG techniques
and provides detailed analysis of responses.
"""

import streamlit as st
import os
import tempfile
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import uuid
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our RAG systems
from adaptive_rag import AdaptiveRAG
from crag import CRAG
from document_augmentation import DocumentProcessor, SentenceTransformerEmbeddings, load_document_content
from helper_functions import encode_document

# Configure page
st.set_page_config(
    page_title="Advanced Multi-RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .comparison-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .technique-column {
        flex: 1;
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .technique-header {
        background: #667eea;
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .response-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        min-height: 150px;
        margin-top: 1rem;
    }
    
    .metrics-box {
        background: linear-gradient(135deg, #667eea20 0%, #764ba240 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .message-comparison {
        background: #f0f7ff;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #e3f2fd;
    }
    
    .timing-badge {
        background: #4caf50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .error-badge {
        background: #f44336;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .analytics-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = False
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'rag_systems' not in st.session_state:
        st.session_state.rag_systems = {}
    if 'response_times' not in st.session_state:
        st.session_state.response_times = {}
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = []

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return path"""
    try:
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
                    return AdaptiveRAG(file_paths=document_paths)
                else:
                    return AdaptiveRAG(texts=["Sample text for testing. This is a basic RAG system."])
            
            elif technique == "CRAG":
                if document_paths and len(document_paths) > 0:
                    return CRAG(document_paths[0])
                else:
                    sample_file = "data/sample_text.txt"
                    if os.path.exists(sample_file):
                        return CRAG(sample_file)
                    else:
                        st.error("CRAG requires a document. Please upload a file first.")
                        return None
            
            elif technique == "Document Augmentation":
                if document_paths and len(document_paths) > 0:
                    content = load_document_content(document_paths[0])
                    embedding_model = SentenceTransformerEmbeddings()
                    processor = DocumentProcessor(content, embedding_model, document_paths[0])
                    return processor.run()
                else:
                    st.error("Document Augmentation requires a document. Please upload a file first.")
                    return None
            
            elif technique == "Basic RAG":
                if document_paths and len(document_paths) > 0:
                    return encode_document(document_paths[0])
                else:
                    st.error("Basic RAG requires a document. Please upload a file first.")
                    return None
                    
    except Exception as e:
        st.error(f"Error loading {technique}: {str(e)}")
        return None

def get_rag_response_with_timing(technique: str, query: str, rag_system) -> Tuple[str, float]:
    """Get response from RAG system with timing information"""
    start_time = time.time()
    
    try:
        if technique == "Adaptive RAG":
            response = rag_system.answer(query)
        
        elif technique == "CRAG":
            response = rag_system.run(query)
        
        elif technique == "Document Augmentation":
            docs = rag_system.get_relevant_documents(query)
            if docs:
                from document_augmentation import generate_answer
                context = docs[0].metadata.get('text', docs[0].page_content)
                response = generate_answer(context, query)
            else:
                response = "No relevant documents found."
        
        elif technique == "Basic RAG":
            docs = rag_system.similarity_search(query, k=3)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                response = f"Based on the documents:\n\n{context[:500]}..."
            else:
                response = "No relevant documents found."
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return response, response_time
                
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        return f"Error generating response: {str(e)}", response_time

def display_comparison_results(query: str, results: Dict[str, Tuple[str, float]]):
    """Display side-by-side comparison of RAG results"""
    st.markdown("### 🔍 RAG Technique Comparison")
    
    # Create columns for each technique
    techniques = list(results.keys())
    cols = st.columns(len(techniques))
    
    for i, (technique, (response, response_time)) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="technique-column">
                <div class="technique-header">{technique}</div>
                <div class="response-box">
                    {response[:500]}{'...' if len(response) > 500 else ''}
                </div>
                <div class="timing-badge">⏱️ {response_time:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Add comparison analytics
    st.markdown("### 📊 Performance Analytics")
    
    # Response time comparison
    fig = go.Figure(data=[
        go.Bar(
            x=list(results.keys()),
            y=[results[tech][1] for tech in results.keys()],
            marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c'][:len(results)]
        )
    ])
    
    fig.update_layout(
        title="Response Time Comparison",
        xaxis_title="RAG Technique",
        yaxis_title="Time (seconds)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Response length comparison
    fig2 = go.Figure(data=[
        go.Bar(
            x=list(results.keys()),
            y=[len(results[tech][0]) for tech in results.keys()],
            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24'][:len(results)]
        )
    ])
    
    fig2.update_layout(
        title="Response Length Comparison",
        xaxis_title="RAG Technique",
        yaxis_title="Characters",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)

def display_analytics_dashboard():
    """Display comprehensive analytics dashboard"""
    st.markdown("### 📈 Analytics Dashboard")
    
    if not st.session_state.messages:
        st.info("No data available yet. Start chatting to see analytics!")
        return
    
    # Prepare data
    technique_usage = {}
    response_times_by_technique = {}
    
    for message in st.session_state.messages:
        if message["role"] == "assistant" and message.get("technique"):
            technique = message["technique"]
            technique_usage[technique] = technique_usage.get(technique, 0) + 1
            
            if technique not in response_times_by_technique:
                response_times_by_technique[technique] = []
            
            # Add mock response time if not available
            response_time = message.get("response_time", 1.5)
            response_times_by_technique[technique].append(response_time)
    
    if technique_usage:
        # Usage pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(technique_usage.keys()),
            values=list(technique_usage.values()),
            hole=0.3
        )])
        
        fig_pie.update_layout(
            title="RAG Technique Usage Distribution",
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Average response times
        if response_times_by_technique:
            avg_times = {tech: sum(times)/len(times) for tech, times in response_times_by_technique.items()}
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=list(avg_times.keys()),
                    y=list(avg_times.values()),
                    marker_color='#667eea'
                )
            ])
            
            fig_bar.update_layout(
                title="Average Response Times",
                xaxis_title="RAG Technique",
                yaxis_title="Average Time (seconds)",
                height=400
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced Multi-RAG Chatbot</h1>
        <p>Compare and analyze different RAG techniques with your documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'csv', 'json', 'docx', 'xlsx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, CSV, JSON, DOCX, XLSX"
        )
        
        if uploaded_files:
            document_paths = []
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    document_paths.append(file_path)
            
            st.session_state.uploaded_documents = document_paths
            st.success(f"✅ Uploaded {len(document_paths)} document(s)")
            
            # Display uploaded files
            for file_path in document_paths:
                st.write(f"📄 {os.path.basename(file_path)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Mode Selection
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🎯 Chat Mode")
        
        mode = st.radio(
            "Choose mode:",
            ["Single Technique", "Compare All Techniques"],
            help="Single: Use one technique at a time. Compare: Test all techniques simultaneously."
        )
        
        st.session_state.comparison_mode = (mode == "Compare All Techniques")
        
        if not st.session_state.comparison_mode:
            # Single technique selection
            rag_techniques = ["Adaptive RAG", "CRAG", "Document Augmentation", "Basic RAG"]
            selected_technique = st.selectbox("Choose RAG technique:", rag_techniques)
            
            # Technique description
            descriptions = {
                "Adaptive RAG": "🧠 Dynamically adapts strategy based on query type",
                "CRAG": "🔍 Self-correcting with web search fallback",
                "Document Augmentation": "📝 Enhanced with synthetic questions",
                "Basic RAG": "⚡ Standard similarity-based retrieval"
            }
            
            st.info(descriptions[selected_technique])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Controls
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🎛️ Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.comparison_results = []
            st.rerun()
        
        if st.button("📊 Show Analytics"):
            st.session_state.show_analytics = not st.session_state.get('show_analytics', False)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    main_col, analytics_col = st.columns([2, 1])
    
    with main_col:
        st.header("💬 Chat Interface")
        
        # Display messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message-comparison">
                    <strong>🙋 You:</strong> {message["content"]}
                    <br><small>⏰ {datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                technique_info = f" using {message['technique']}" if message.get('technique') else ""
                timing_info = f" ({message.get('response_time', 0):.2f}s)" if message.get('response_time') else ""
                
                st.markdown(f"""
                <div class="message-comparison">
                    <strong>🤖 Assistant{technique_info}{timing_info}:</strong>
                    <br>{message["content"]}
                    <br><small>⏰ {datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            # Add user message
            user_message = {
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            
            if st.session_state.comparison_mode:
                # Compare all techniques
                st.info("🔄 Running comparison across all RAG techniques...")
                
                techniques = ["Adaptive RAG", "CRAG", "Document Augmentation", "Basic RAG"]
                results = {}
                
                for technique in techniques:
                    # Load RAG system if not already loaded
                    if technique not in st.session_state.rag_systems:
                        rag_system = load_rag_system(technique, st.session_state.uploaded_documents)
                        if rag_system:
                            st.session_state.rag_systems[technique] = rag_system
                        else:
                            continue
                    
                    # Get response with timing
                    rag_system = st.session_state.rag_systems[technique]
                    response, response_time = get_rag_response_with_timing(technique, query, rag_system)
                    results[technique] = (response, response_time)
                    
                    # Add individual message
                    assistant_message = {
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": response,
                        "technique": technique,
                        "response_time": response_time,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(assistant_message)
                
                # Display comparison
                if results:
                    display_comparison_results(query, results)
                    st.session_state.comparison_results.append({
                        "query": query,
                        "results": results,
                        "timestamp": datetime.now().isoformat()
                    })
            
            else:
                # Single technique mode
                # Load RAG system if not already loaded
                if selected_technique not in st.session_state.rag_systems:
                    rag_system = load_rag_system(selected_technique, st.session_state.uploaded_documents)
                    if rag_system:
                        st.session_state.rag_systems[selected_technique] = rag_system
                    else:
                        st.error(f"Failed to load {selected_technique}")
                        st.stop()
                
                # Get response
                rag_system = st.session_state.rag_systems[selected_technique]
                response, response_time = get_rag_response_with_timing(selected_technique, query, rag_system)
                
                # Add assistant message
                assistant_message = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": response,
                    "technique": selected_technique,
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(assistant_message)
            
            st.rerun()
    
    with analytics_col:
        st.header("📊 Analytics")
        
        # Quick stats
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        
        st.markdown(f"""
        <div class="analytics-card">
            <h4>📈 Quick Stats</h4>
            <p><strong>Total Messages:</strong> {total_messages}</p>
            <p><strong>Questions Asked:</strong> {user_messages}</p>
            <p><strong>Documents:</strong> {len(st.session_state.uploaded_documents)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recent activity
        if st.session_state.messages:
            recent_messages = st.session_state.messages[-5:]
            st.markdown("""
            <div class="analytics-card">
                <h4>🕒 Recent Activity</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for msg in recent_messages:
                if msg["role"] == "assistant":
                    technique = msg.get("technique", "Unknown")
                    time_str = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M")
                    st.write(f"• {time_str} - {technique}")
        
        # Export options
        if st.session_state.messages:
            st.markdown("""
            <div class="analytics-card">
                <h4>💾 Export Options</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📥 Download Chat History"):
                chat_data = {
                    "messages": st.session_state.messages,
                    "comparison_results": st.session_state.comparison_results,
                    "export_time": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(chat_data, indent=2),
                    file_name=f"advanced_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Analytics dashboard (full width)
    if st.session_state.get('show_analytics', False):
        st.markdown("---")
        display_analytics_dashboard()

if __name__ == "__main__":
    main()
