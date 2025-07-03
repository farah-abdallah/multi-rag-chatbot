"""
Streamlit UI components for the Multi-RAG Chatbot.
"""
import streamlit as st
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from ..utils.logging import get_logger

logger = get_logger(__name__)


def render_sidebar() -> Optional[str]:
    """Render the sidebar with navigation and controls."""
    with st.sidebar:
        st.title("🤖 Multi-RAG Chatbot")
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        
        action = None
        
        # Document management
        if st.button("📁 Upload Documents", use_container_width=True):
            action = "upload_documents"
        
        if st.button("🗑️ Clear Documents", use_container_width=True):
            action = "clear_documents"
        
        st.markdown("---")
        
        # Settings and info
        if st.button("⚙️ Settings", use_container_width=True):
            action = "show_settings"
        
        if st.button("📊 System Stats", use_container_width=True):
            action = "show_stats"
        
        st.markdown("---")
        
        # System info
        st.subheader("System Info")
        st.info("Multi-RAG Chatbot v2.0")
        st.info("Ready for questions")
        
        # Quick help
        with st.expander("Quick Help"):
            st.markdown("""
            **How to use:**
            1. Upload documents using the "Upload Documents" button
            2. Ask questions in the chat interface
            3. View source highlights and references
            4. Adjust settings as needed
            
            **Supported formats:**
            - PDF, DOCX, PPTX
            - TXT, MD, CSV
            - XLSX, XLS, JSON
            """)
        
        return action


def render_file_uploader(accepted_types: List[str] = None) -> List[Any]:
    """Render file uploader component."""
    if accepted_types is None:
        accepted_types = ['pdf', 'docx', 'pptx', 'txt', 'md', 'csv', 'xlsx', 'xls', 'json']
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=accepted_types,
        help="Upload documents to add to the knowledge base"
    )
    
    return uploaded_files or []


def render_chat_interface() -> Dict[str, Any]:
    """Render the chat interface."""
    st.subheader("💬 Chat Interface")
    
    # Chat input
    query = st.text_input(
        "Ask a question:",
        placeholder="Enter your question here...",
        key="chat_input"
    )
    
    # Send button
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        send_clicked = st.button("Send", disabled=st.session_state.get('processing', False))
    
    with col2:
        clear_clicked = st.button("Clear Chat")
    
    return {
        'query': query,
        'send_clicked': send_clicked,
        'clear_clicked': clear_clicked
    }


def render_chat_history(chat_history: List[Dict[str, Any]]):
    """Render chat history."""
    for i, message in enumerate(chat_history):
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(message['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(message['content'])
                
                # Show metadata if available
                if 'metadata' in message:
                    metadata = message['metadata']
                    if metadata.get('confidence'):
                        st.caption(f"Confidence: {metadata['confidence']:.2f}")
                    if metadata.get('method_used'):
                        st.caption(f"Method: {metadata['method_used']}")


def render_source_highlights(highlights: List[Any]):
    """Render source highlights."""
    if not highlights:
        return
    
    st.subheader("🔍 Source Highlights")
    
    for i, highlight in enumerate(highlights[:5]):  # Show top 5
        with st.expander(f"{highlight.document_name} (Score: {highlight.relevance_score:.2f})"):
            st.markdown(f"**Document:** {highlight.document_name}")
            st.markdown(f"**Relevance Score:** {highlight.relevance_score:.2f}")
            st.markdown(f"**Highlighted Text:**")
            st.markdown(f"*{highlight.highlighted_text}*")
            
            if highlight.context_before or highlight.context_after:
                st.markdown(f"**Context:**")
                context = f"...{highlight.context_before}**{highlight.highlighted_text}**{highlight.context_after}..."
                st.markdown(context)
            
            if highlight.page_number:
                st.markdown(f"**Page:** {highlight.page_number}")
            
            if highlight.section:
                st.markdown(f"**Section:** {highlight.section}")


def render_document_summary(documents: List[Dict[str, Any]]):
    """Render document summary."""
    if not documents:
        st.info("No documents uploaded yet. Upload documents to get started.")
        return
    
    st.subheader("📚 Document Summary")
    
    # Summary statistics
    total_docs = len(documents)
    total_size = sum(doc.get('metadata', {}).get('content_length', 0) for doc in documents)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Content", f"{total_size:,} chars")
    
    # Document list
    with st.expander("Document Details"):
        for doc in documents:
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            file_size = metadata.get('file_size', 0)
            content_length = metadata.get('content_length', 0)
            loaded_at = metadata.get('loaded_at', 'Unknown')
            
            st.markdown(f"**{filename}**")
            st.markdown(f"- File size: {file_size:,} bytes")
            st.markdown(f"- Content length: {content_length:,} characters")
            st.markdown(f"- Loaded at: {loaded_at}")
            st.markdown("---")


def render_system_stats(stats: Dict[str, Any]):
    """Render system statistics."""
    st.subheader("📊 System Statistics")
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        retrieval_stats = stats.get('retrieval', {})
        st.metric("Total Documents", retrieval_stats.get('total_documents', 0))
    
    with col2:
        llm_stats = stats.get('llm', {})
        st.metric("API Calls", llm_stats.get('total_usage', 0))
    
    with col3:
        evaluation_stats = stats.get('evaluation', {})
        avg_score = evaluation_stats.get('average_scores', {}).get('overall', 0)
        st.metric("Avg Quality Score", f"{avg_score:.2f}")
    
    # Detailed stats
    with st.expander("Detailed Statistics"):
        st.json(stats)


def render_settings(current_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Render settings interface."""
    st.subheader("⚙️ Settings")
    
    new_settings = current_settings.copy()
    
    # Web search settings
    st.markdown("**Web Search**")
    new_settings['use_web_search'] = st.checkbox(
        "Enable Web Search",
        value=current_settings.get('use_web_search', True),
        help="Enable web search to supplement document knowledge"
    )
    
    new_settings['max_search_results'] = st.slider(
        "Max Search Results",
        min_value=1,
        max_value=10,
        value=current_settings.get('max_search_results', 5),
        help="Maximum number of web search results to consider"
    )
    
    st.markdown("---")
    
    # Document processing settings
    st.markdown("**Document Processing**")
    new_settings['chunk_size'] = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=2000,
        value=current_settings.get('chunk_size', 1000),
        step=100,
        help="Size of document chunks for processing"
    )
    
    new_settings['chunk_overlap'] = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=current_settings.get('chunk_overlap', 200),
        step=50,
        help="Overlap between adjacent chunks"
    )
    
    st.markdown("---")
    
    # Display settings
    st.markdown("**Display**")
    new_settings['show_source_highlights'] = st.checkbox(
        "Show Source Highlights",
        value=current_settings.get('show_source_highlights', True),
        help="Show highlighted sources in responses"
    )
    
    new_settings['show_system_stats'] = st.checkbox(
        "Show System Statistics",
        value=current_settings.get('show_system_stats', False),
        help="Show system performance statistics"
    )
    
    new_settings['show_confidence_scores'] = st.checkbox(
        "Show Confidence Scores",
        value=current_settings.get('show_confidence_scores', True),
        help="Show confidence scores for responses"
    )
    
    return new_settings


def render_error_message(error: str):
    """Render error message."""
    st.error(f"❌ Error: {error}")


def render_success_message(message: str):
    """Render success message."""
    st.success(f"✅ {message}")


def render_info_message(message: str):
    """Render info message."""
    st.info(f"ℹ️ {message}")


def render_warning_message(message: str):
    """Render warning message."""
    st.warning(f"⚠️ {message}")


def render_progress_bar(progress: float, text: str = ""):
    """Render progress bar."""
    st.progress(progress, text=text)


def render_spinner(text: str = "Loading..."):
    """Render spinner context manager."""
    return st.spinner(text)


def render_expander(title: str, expanded: bool = False):
    """Render expander context manager."""
    return st.expander(title, expanded=expanded)


def render_tabs(tab_names: List[str]):
    """Render tabs."""
    return st.tabs(tab_names)


def render_columns(sizes: List[int]):
    """Render columns."""
    return st.columns(sizes)


def render_container():
    """Render container."""
    return st.container()


def render_empty():
    """Render empty placeholder."""
    return st.empty()


def render_json_viewer(data: Dict[str, Any], label: str = "Data"):
    """Render JSON data viewer."""
    with st.expander(f"📋 {label}"):
        st.json(data)


def render_code_block(code: str, language: str = "python"):
    """Render code block."""
    st.code(code, language=language)


def render_markdown(content: str):
    """Render markdown content."""
    st.markdown(content, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, delta: str = None):
    """Render metric card."""
    st.metric(title, value, delta)


def render_chart_placeholder():
    """Render chart placeholder."""
    st.info("Chart functionality not implemented yet")


def render_download_button(data: str, filename: str, mime_type: str = "text/plain"):
    """Render download button."""
    return st.download_button(
        label=f"Download {filename}",
        data=data,
        file_name=filename,
        mime=mime_type
    )


def render_confirmation_dialog(message: str, key: str) -> bool:
    """Render confirmation dialog."""
    return st.button(message, key=key, type="primary")


def render_loading_state(is_loading: bool, message: str = "Processing..."):
    """Render loading state."""
    if is_loading:
        st.spinner(message)
    else:
        st.empty()


def render_status_indicator(status: str, message: str = ""):
    """Render status indicator."""
    if status == "success":
        st.success(message or "Operation completed successfully")
    elif status == "error":
        st.error(message or "An error occurred")
    elif status == "warning":
        st.warning(message or "Warning")
    elif status == "info":
        st.info(message or "Information")
    else:
        st.write(message)


def render_key_value_pairs(data: Dict[str, Any], title: str = "Details"):
    """Render key-value pairs."""
    st.subheader(title)
    for key, value in data.items():
        st.write(f"**{key}:** {value}")


def render_list_items(items: List[str], title: str = "Items"):
    """Render list items."""
    st.subheader(title)
    for item in items:
        st.write(f"• {item}")


def render_search_box(placeholder: str = "Search...") -> str:
    """Render search box."""
    return st.text_input("", placeholder=placeholder, key="search_box")


def render_filter_controls(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Render filter controls."""
    st.subheader("🔍 Filters")
    
    updated_filters = {}
    
    for filter_name, filter_config in filters.items():
        filter_type = filter_config.get('type', 'text')
        filter_label = filter_config.get('label', filter_name)
        filter_value = filter_config.get('value', '')
        
        if filter_type == 'text':
            updated_filters[filter_name] = st.text_input(filter_label, value=filter_value)
        elif filter_type == 'select':
            options = filter_config.get('options', [])
            updated_filters[filter_name] = st.selectbox(filter_label, options, index=options.index(filter_value) if filter_value in options else 0)
        elif filter_type == 'multiselect':
            options = filter_config.get('options', [])
            updated_filters[filter_name] = st.multiselect(filter_label, options, default=filter_value)
        elif filter_type == 'slider':
            min_val = filter_config.get('min', 0)
            max_val = filter_config.get('max', 100)
            updated_filters[filter_name] = st.slider(filter_label, min_val, max_val, filter_value)
        elif filter_type == 'checkbox':
            updated_filters[filter_name] = st.checkbox(filter_label, value=filter_value)
    
    return updated_filters
