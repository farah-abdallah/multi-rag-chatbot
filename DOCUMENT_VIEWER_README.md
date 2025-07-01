# Document Viewer Integration for CRAG

This implementation adds advanced document viewing capabilities to the CRAG (Corrective RAG) system, allowing users to see exactly which parts of their documents were used to generate answers.

## Features Added

### 1. Enhanced CRAG with Source Tracking
- **New Method**: `run_with_sources()` returns both answer and source chunk information
- **Source Metadata**: Tracks document paths, page numbers, paragraph numbers, and relevance scores
- **Backward Compatibility**: Original `run()` method still works unchanged

### 2. Document Viewer Component
- **Highlighting**: Shows retrieved text chunks highlighted in different colors
- **Multiple View Options**: 
  - Open in new tab for full document view
  - Embedded viewer within the chat interface
- **Navigation**: Search within documents using Ctrl+F
- **Download**: Option to download original documents

### 3. Streamlit Integration
- **Automatic Detection**: App detects when CRAG finds relevant document chunks
- **Source Links**: Clickable buttons to view source documents
- **Chunk Information**: Shows relevance scores and metadata for each chunk
- **Visual Feedback**: Color-coded highlights for different text chunks

## How It Works

### For Users
1. **Upload Documents**: Use any supported format (PDF, TXT, CSV, JSON, DOCX, XLSX)
2. **Select CRAG**: Choose CRAG as your RAG technique
3. **Ask Questions**: Submit your query as normal
4. **View Sources**: Click the document viewer links that appear with answers
5. **Explore**: See exactly which parts of your documents contributed to the answer

### For Developers
1. **CRAG Enhancement**: The `CRAG` class now tracks source chunks during processing
2. **Document Viewer**: New `document_viewer.py` module handles document display
3. **Chatbot Integration**: Updated `chatbot_app.py` to use enhanced CRAG features

## File Changes Made

### New Files
- `document_viewer.py` - Document viewing and highlighting functionality
- `test_document_viewer_integration.py` - Test script for new features

### Modified Files
- `crag.py` - Added source tracking and `run_with_sources()` method
- `chatbot_app.py` - Integrated document viewer with CRAG responses
- `requirements.txt` - Added beautifulsoup4 and requests dependencies

## Usage Examples

### Enhanced CRAG Usage
```python
from crag import CRAG

# Initialize CRAG
crag_system = CRAG(file_path="document.pdf")

# Get answer with source information
result = crag_system.run_with_sources("What is the main topic?")

print(f"Answer: {result['answer']}")
print(f"Source chunks: {len(result['source_chunks'])}")
for chunk in result['source_chunks']:
    print(f"  - Score: {chunk['score']}")
    print(f"  - Text: {chunk['text'][:100]}...")
```

### Document Viewer Usage
```python
from document_viewer import show_embedded_document_viewer

# Show document with highlighted chunks
chunks = [{'text': 'relevant text chunk', 'score': 0.85}]
show_embedded_document_viewer("document.pdf", chunks)
```

## Requirements

### New Dependencies
- `beautifulsoup4` - For HTML parsing in web search
- `requests` - For web search functionality

### Existing Dependencies
All existing dependencies remain the same.

## Testing

Run the integration test to verify everything works:

```bash
python test_document_viewer_integration.py
```

## Benefits

1. **Transparency**: Users can see exactly which parts of their documents were used
2. **Verification**: Easy to verify if the AI's answer is based on correct source material
3. **Trust**: Builds confidence in AI responses through source visibility
4. **Navigation**: Quick access to original document context
5. **Research**: Helps users explore related information in source documents

## Compatibility

- **Backward Compatible**: All existing CRAG functionality preserved
- **Optional Features**: Document viewer features are optional enhancements
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Browser Support**: Works with all modern web browsers through Streamlit

## Future Enhancements

Possible future improvements:
- Support for highlighting in PDF files directly
- Document annotations and bookmarking
- Export of highlighted documents
- Integration with other RAG techniques beyond CRAG
- Multi-document comparison views
