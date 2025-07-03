"""
Custom styles for the Multi-RAG Chatbot Streamlit interface.
"""
import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown("""
    <style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .sidebar .sidebar-content .element-container {
        margin-bottom: 1rem;
    }
    
    /* Chat interface styling */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }
    
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .chat-message.assistant {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    /* Source highlight styling */
    .source-highlight {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .source-highlight .document-name {
        font-weight: bold;
        color: #856404;
    }
    
    .source-highlight .relevance-score {
        color: #28a745;
        font-weight: bold;
    }
    
    .source-highlight .highlighted-text {
        background-color: #ffeb3b;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-style: italic;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-card .metric-title {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #495057;
    }
    
    /* Status indicators */
    .status-success {
        color: #28a745;
    }
    
    .status-error {
        color: #dc3545;
    }
    
    .status-warning {
        color: #ffc107;
    }
    
    .status-info {
        color: #17a2b8;
    }
    
    /* Document summary */
    .document-summary {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .document-summary .document-item {
        border-bottom: 1px solid #e9ecef;
        padding: 0.5rem 0;
    }
    
    .document-summary .document-item:last-child {
        border-bottom: none;
    }
    
    /* Settings panel */
    .settings-panel {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .settings-panel .setting-group {
        margin-bottom: 1.5rem;
    }
    
    .settings-panel .setting-group h4 {
        color: #495057;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Statistics display */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stats-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stats-card .stats-title {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    
    .stats-card .stats-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #495057;
    }
    
    /* Loading states */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #007bff;
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #0056b3;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #007bff;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    .stFileUploader:hover {
        border-color: #0056b3;
        background-color: #e3f2fd;
    }
    
    /* Expander */
    .streamlit-expander {
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .streamlit-expander .streamlit-expanderHeader {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    
    .streamlit-expander .streamlit-expanderContent {
        padding: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
        background-color: #f8f9fa;
        border: 1px solid #e1e5e9;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    
    /* Columns spacing */
    .stColumns {
        gap: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #007bff;
        border-radius: 0.5rem;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 0.5rem;
    }
    
    /* Text input */
    .stTextInput > div > div {
        border-radius: 0.5rem;
    }
    
    /* Textarea */
    .stTextArea > div > div {
        border-radius: 0.5rem;
    }
    
    /* Slider */
    .stSlider > div > div {
        border-radius: 0.5rem;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        font-weight: 500;
    }
    
    /* Radio */
    .stRadio > label {
        font-weight: 500;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        border-radius: 0.5rem;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
    
    /* Error message */
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
    
    /* Warning message */
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
    
    /* Info message */
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
    
    /* Code block */
    .stCodeBlock {
        background-color: #f8f9fa;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        font-family: 'Courier New', monospace;
    }
    
    /* JSON viewer */
    .stJson {
        background-color: #f8f9fa;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    /* Metric */
    .stMetric {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stMetric .metric-container {
        text-align: center;
    }
    
    .stMetric .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    
    .stMetric .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #495057;
    }
    
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    
    .sidebar .sidebar-content .element-container {
        margin-bottom: 0.5rem;
    }
    
    .sidebar .sidebar-content h1 {
        color: #495057;
        margin-bottom: 1rem;
    }
    
    .sidebar .sidebar-content h2 {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #007bff;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #0056b3;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .stats-container {
            grid-template-columns: 1fr;
        }
        
        .stColumns {
            flex-direction: column;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main .block-container {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        .sidebar .sidebar-content {
            background-color: #2d2d2d;
        }
        
        .chat-message {
            background-color: #3d3d3d;
            border-color: #555555;
        }
        
        .metric-card {
            background-color: #3d3d3d;
            color: #ffffff;
        }
        
        .stats-card {
            background-color: #3d3d3d;
            color: #ffffff;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def get_theme_colors():
    """Get theme colors for consistent styling."""
    return {
        'primary': '#007bff',
        'secondary': '#6c757d',
        'success': '#28a745',
        'danger': '#dc3545',
        'warning': '#ffc107',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40',
        'white': '#ffffff',
        'black': '#000000'
    }


def get_font_sizes():
    """Get font sizes for consistent typography."""
    return {
        'xs': '0.75rem',
        'sm': '0.875rem',
        'base': '1rem',
        'lg': '1.125rem',
        'xl': '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem',
        '4xl': '2.25rem'
    }


def get_spacing():
    """Get spacing values for consistent layout."""
    return {
        'xs': '0.25rem',
        'sm': '0.5rem',
        'base': '1rem',
        'lg': '1.5rem',
        'xl': '2rem',
        '2xl': '3rem',
        '3xl': '4rem'
    }


def get_border_radius():
    """Get border radius values for consistent styling."""
    return {
        'none': '0',
        'sm': '0.125rem',
        'base': '0.25rem',
        'md': '0.375rem',
        'lg': '0.5rem',
        'xl': '0.75rem',
        '2xl': '1rem',
        'full': '9999px'
    }


def get_shadows():
    """Get shadow values for consistent depth."""
    return {
        'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'base': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
    }


def apply_component_style(component_type: str, custom_styles: dict = None):
    """Apply styles to specific components."""
    styles = custom_styles or {}
    
    if component_type == 'chat_message':
        return f"""
        <style>
        .chat-message {{
            padding: {styles.get('padding', '1rem')};
            margin: {styles.get('margin', '0.5rem 0')};
            border-radius: {styles.get('border_radius', '0.5rem')};
            background-color: {styles.get('background_color', '#f8f9fa')};
            border-left: {styles.get('border_left', '4px solid #007bff')};
        }}
        </style>
        """
    
    elif component_type == 'metric_card':
        return f"""
        <style>
        .metric-card {{
            background-color: {styles.get('background_color', '#ffffff')};
            padding: {styles.get('padding', '1rem')};
            border-radius: {styles.get('border_radius', '0.5rem')};
            box-shadow: {styles.get('box_shadow', '0 2px 4px rgba(0,0,0,0.1)')};
            margin: {styles.get('margin', '0.5rem 0')};
        }}
        </style>
        """
    
    elif component_type == 'source_highlight':
        return f"""
        <style>
        .source-highlight {{
            background-color: {styles.get('background_color', '#fff3cd')};
            border: {styles.get('border', '1px solid #ffeaa7')};
            border-radius: {styles.get('border_radius', '0.25rem')};
            padding: {styles.get('padding', '0.5rem')};
            margin: {styles.get('margin', '0.5rem 0')};
        }}
        </style>
        """
    
    return ""


def create_custom_css(styles: dict):
    """Create custom CSS from a dictionary of styles."""
    css_rules = []
    
    for selector, properties in styles.items():
        rule = f"{selector} {{\n"
        for prop, value in properties.items():
            rule += f"    {prop}: {value};\n"
        rule += "}\n"
        css_rules.append(rule)
    
    return f"<style>\n{''.join(css_rules)}</style>"
