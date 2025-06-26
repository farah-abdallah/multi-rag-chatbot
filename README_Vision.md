# 🔍 Vision-Enhanced Multi-RAG Chatbot

Advanced document analysis system with **Vision LLM capabilities** that can read and understand images, charts, graphs, and diagrams from your documents.

## 🚀 What's New: Vision Capabilities

Your Multi-RAG system now includes **Vision LLM** integration that can:

- 📊 **Extract data from charts and graphs** with specific data points and trends
- 🗺️ **Analyze maps, diagrams, and flowcharts** with relationship understanding  
- 📋 **Read tables and infographics** from images
- 🔍 **Process complex visual content** and integrate it with text analysis
- 📄 **Automatically analyze PDF images** during document processing

## 🛠️ Installation

### Quick Setup
```bash
# Run the automated installation script
powershell -ExecutionPolicy Bypass -File install_vision_requirements.ps1
```

### Manual Installation
```bash
# Core dependencies
pip install langchain langchain-google-genai langchain-community faiss-cpu python-dotenv sentence-transformers streamlit pandas

# Vision processing (NEW)
pip install PyMuPDF Pillow

# Optional enhancements
pip install unstructured duckduckgo-search requests
```

### Environment Setup
Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

## 🎯 How to Use

### 1. Run the Vision-Enhanced Chatbot
```bash
streamlit run vision_enhanced_chatbot.py
```

### 2. Test Individual RAG Techniques
```bash
# Test Adaptive RAG with vision
python adaptive_rag.py --files "document_with_images.pdf" --query "What do the charts show?"

# Test CRAG with vision awareness  
python crag.py --file_path "document_with_images.pdf" --query "Analyze the data trends"

# Run comprehensive demo
python vision_demo.py --file "your_document.pdf" --query "What insights are shown in the visualizations?"
```

## 🔍 Vision Features by RAG Technique

### 🧠 Adaptive RAG + Vision
- **Intelligent query classification** with vision-aware context
- **Adaptive retrieval strategies** for visual content
- **Semantic understanding** of both text and images
- **Best for**: Educational content, mixed-media documents

### 🔧 CRAG + Vision  
- **Self-correcting retrieval** with vision quality assessment
- **Web search fallback** enhanced with visual context
- **Balanced accuracy** across text and visual content
- **Best for**: Research papers, technical documents

### 📄 Document Augmentation + Vision
- **Synthetic Q&A generation** from visual content
- **Enhanced semantic search** with image-derived questions
- **Fast processing** with vision content integration
- **Best for**: Large document collections, data visualization

### 💡 Explainable Retrieval + Vision
- **Transparent reasoning** for visual content selection
- **Source attribution** linking insights to specific images
- **Comprehensive explanations** of visual relevance
- **Best for**: Scientific papers, analytical reports

## 📊 Example Use Cases

### Climate Change Analysis
```python
# Upload: Understanding_Climate_Change.pdf (with charts and graphs)
query = "What temperature trends are shown in the data visualizations?"

# Vision capabilities extract:
# - Temperature graph data points
# - Trend analysis from charts  
# - Geographic patterns from maps
# - Emission data from infographics
```

### Business Report Analysis
```python
# Upload: quarterly_report.pdf (with financial charts)
query = "What do the financial performance charts indicate?"

# Vision capabilities extract:
# - Revenue trends from line graphs
# - Market share from pie charts
# - Performance metrics from bar charts
# - Comparative analysis from tables
```

### Technical Documentation  
```python
# Upload: system_architecture.pdf (with diagrams)
query = "How does the system architecture work?"

# Vision capabilities extract:
# - Process flows from diagrams
# - Component relationships
# - Data flow patterns
# - System connections
```

## 🎮 Demo Commands

### Basic Vision Demo
```bash
python vision_demo.py --file "Understanding_Climate_Change.pdf" --query "What data trends are shown?"
```

### Comprehensive Analysis
```bash
python vision_demo.py --file "your_document.pdf" --query "Analyze all visual content and data"
```

### Performance Comparison
```bash
# The demo automatically compares all 4 techniques with timing and quality metrics
```

## 📋 Supported File Formats

| Format | Text Extraction | Vision Analysis | Use Case |
|--------|----------------|-----------------|----------|
| **PDF** | ✅ | ✅ **NEW** | Research papers, reports with charts |
| **DOCX** | ✅ | ⚠️ Limited | Business documents |
| **TXT** | ✅ | ❌ | Plain text documents |
| **CSV** | ✅ | ❌ | Data files |
| **JSON** | ✅ | ❌ | Structured data |
| **XLSX** | ✅ | ❌ | Spreadsheets |

## 🔧 Configuration Options

### Vision Processing Settings
```python
# In adaptive_rag.py, document_augmentation.py, etc.

# Enable/disable vision processing
VISION_AVAILABLE = True  # Automatically detected

# Vision analysis prompts (customizable)
vision_prompt = """Analyze this image and extract:
- Key data points and trends
- Relationships and patterns  
- Numerical information
- Important insights"""
```

### Performance Optimization
```python
# Adjust chunk sizes for better vision integration
chunk_size = 1000        # Text chunk size
chunk_overlap = 200      # Overlap for context
vision_max_images = 10   # Max images to process per document
```

## 📊 Performance Expectations

| Technique | Speed | Vision Processing | Best For |
|-----------|-------|------------------|----------|
| **Adaptive RAG** | Medium | ✅ Full analysis | Mixed content types |
| **CRAG** | Fast | ✅ Quality-aware | Accuracy-critical |
| **Doc Augmentation** | Fastest | ✅ Efficient | High-volume processing |
| **Explainable** | Slower | ✅ Detailed | Research/analysis |

## 🛠️ Troubleshooting

### Vision Processing Issues
```bash
# Install vision dependencies
pip install PyMuPDF Pillow

# Test vision availability
python -c "import fitz, PIL.Image; print('✅ Vision ready')"
```

### API Issues
```bash
# Check API key
echo $GOOGLE_API_KEY

# Test Gemini connection
python -c "import google.generativeai as genai; genai.configure(api_key='your_key'); print('✅ API ready')"
```

### Performance Issues
```bash
# For large PDFs with many images, adjust settings:
# - Reduce vision_max_images
# - Increase chunk_size  
# - Process documents in batches
```

## 🎯 For Your Mentor Meeting

**Key Points to Highlight:**

1. **Advanced Vision Integration**: All 4 RAG techniques now support image analysis
2. **Automatic PDF Processing**: Charts, graphs, and diagrams are automatically analyzed
3. **Comprehensive Coverage**: Both text and visual content integrated seamlessly
4. **Performance Optimized**: Intelligent processing with configurable settings
5. **Production Ready**: Full error handling, logging, and user interface

**Demo Script:**
```bash
# Show the comprehensive vision demo
python vision_demo.py --file "Understanding_Climate_Change.pdf" --query "What climate data is shown in the visualizations?"

# Launch the interactive chatbot
streamlit run vision_enhanced_chatbot.py
```

This enhancement makes your Multi-RAG system truly **cutting-edge** for processing scientific documents, business reports, and any content with visual data! 🚀
