# 🤖 Multi-RAG Chatbot

A comprehensive chatbot application that allows you to compare different RAG (Retrieval-Augmented Generation) techniques with your own documents. Upload documents, choose from multiple RAG techniques, and see how each approach responds to your questions.

## ✨ Features

### 🔧 Multiple RAG Techniques
- **Adaptive RAG**: Dynamically adapts retrieval strategy based on query type (Factual, Analytical, Opinion, Contextual)
- **CRAG (Corrective RAG)**: Evaluates retrieved documents and falls back to web search if needed
- **Document Augmentation**: Enhances documents with generated questions for better retrieval
- **Basic RAG**: Standard similarity-based retrieval and response generation

### 📁 Multi-Format Document Support
- PDF documents
- Text files (.txt)
- CSV data files
- JSON documents
- Word documents (.docx)
- Excel files (.xlsx)

### 💬 Interactive Chat Interface
- Real-time chat with different RAG techniques
- Message history with technique tracking
- Export chat history to JSON
- Elegant, responsive UI design

### 📊 Analytics & Statistics
- Message count tracking
- Technique usage statistics
- Document loading status
- Chat history export

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google API Key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone or download the project files**

2. **Set up your Google API Key**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env file and add your API key
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_chatbot.txt
   ```

4. **Launch the chatbot**
   
   **Option A: Using the Python launcher**
   ```bash
   python launch_chatbot.py
   ```
   
   **Option B: Using PowerShell (Windows)**
   ```powershell
   .\launch_chatbot.ps1
   ```
   
   **Option C: Direct Streamlit command**
   ```bash
   streamlit run chatbot_app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📖 How to Use

### 1. Upload Documents
- Click "Browse files" in the sidebar
- Select one or more documents (PDF, TXT, CSV, JSON, DOCX, XLSX)
- Wait for the upload confirmation

### 2. Choose RAG Technique
- Select your preferred RAG technique from the dropdown menu
- Each technique has a description explaining its approach

### 3. Ask Questions
- Type your question in the chat input at the bottom
- Press Enter or click Send
- The system will process your question using the selected RAG technique

### 4. View Results
- See the response with the RAG technique badge
- Check the statistics panel for usage information
- Export your chat history if needed

## 🔧 RAG Techniques Explained

### Adaptive RAG
- **Purpose**: Automatically selects the best retrieval strategy based on query type
- **Best for**: Mixed question types, general-purpose usage
- **How it works**: Classifies queries into Factual, Analytical, Opinion, or Contextual categories and applies specialized retrieval strategies

### CRAG (Corrective RAG)
- **Purpose**: Self-correcting RAG that evaluates retrieval quality
- **Best for**: Situations where document coverage might be incomplete
- **How it works**: Evaluates retrieved documents; if quality is low, performs web search as fallback

### Document Augmentation
- **Purpose**: Enhances documents with synthetic questions for better retrieval
- **Best for**: Improving search precision and recall
- **How it works**: Generates questions from document content and uses them to improve retrieval

### Basic RAG
- **Purpose**: Standard similarity-based retrieval
- **Best for**: Simple, straightforward question-answering
- **How it works**: Uses vector similarity search to find relevant document chunks

## 📊 File Format Support

| Format | Extension | Description |
|--------|-----------|-------------|
| PDF | .pdf | Portable Document Format |
| Text | .txt | Plain text files |
| CSV | .csv | Comma-separated values |
| JSON | .json | JavaScript Object Notation |
| Word | .docx | Microsoft Word documents |
| Excel | .xlsx | Microsoft Excel spreadsheets |

## 🎨 UI Features

### Chat Interface
- Clean, modern chat interface
- Color-coded messages (user vs. assistant)
- Technique badges showing which RAG method was used
- Timestamps for all messages

### Sidebar Controls
- Document upload area with drag-and-drop support
- RAG technique selector with descriptions
- Clear history button
- File upload status

### Statistics Panel
- Total message count
- Questions asked counter
- Documents loaded indicator
- Technique usage breakdown
- Chat export functionality

## 🔒 Environment Variables

Create a `.env` file in the project root:

```env
# Required: Your Google API Key
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Custom settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

## 🛠️ Development

### Project Structure
```
├── chatbot_app.py              # Main Streamlit application
├── adaptive_rag.py             # Adaptive RAG implementation
├── crag.py                     # CRAG implementation  
├── document_augmentation.py    # Document augmentation RAG
├── helper_functions.py         # Utility functions
├── requirements_chatbot.txt    # Python dependencies
├── launch_chatbot.py          # Python launcher script
├── launch_chatbot.ps1         # PowerShell launcher script
├── .env                       # Environment variables
└── data/                      # Sample documents
    ├── sample_text.txt
    ├── sample_products.csv
    └── sample_company.json
```

### Adding New RAG Techniques

1. Implement your RAG class in a new Python file
2. Add import to `chatbot_app.py`
3. Add technique to the `rag_techniques` list
4. Add description to `technique_descriptions`
5. Implement loading logic in `load_rag_system()`
6. Implement response logic in `get_rag_response()`

### Customizing the UI

The application uses custom CSS for styling. Modify the CSS in the `st.markdown()` calls in `chatbot_app.py` to customize:
- Colors and themes
- Message bubble styles
- Card layouts
- Typography

## 🚨 Troubleshooting

### Common Issues

**"GOOGLE_API_KEY not found"**
- Make sure you have a `.env` file with your API key
- Verify the API key is valid and active

**"No module named 'streamlit'"**
- Install requirements: `pip install -r requirements_chatbot.txt`

**"Error loading documents"**
- Check file format is supported
- Ensure files are not corrupted
- Try with smaller files first

**"Rate limit exceeded"**
- Wait a few moments between requests
- Check your Google API quota limits

**"Port already in use"**
- Change the port in launch command: `streamlit run chatbot_app.py --server.port 8502`

### Performance Tips

1. **Large Documents**: Break very large documents into smaller chunks
2. **Multiple Files**: Upload files gradually to avoid memory issues
3. **API Limits**: Be mindful of Google API rate limits
4. **Browser Performance**: Clear browser cache if UI becomes slow

## 📚 Dependencies

### Core Requirements
- `streamlit` - Web application framework
- `langchain` - LLM application framework
- `langchain-google-genai` - Google Gemini integration
- `faiss-cpu` - Vector similarity search
- `python-dotenv` - Environment variable management

### Document Processing
- `PyPDF2` - PDF document processing
- `python-docx` - Word document processing
- `openpyxl` - Excel file processing
- `unstructured` - Multi-format document loader
- `sentence-transformers` - Text embeddings

### Additional Features
- `duckduckgo-search` - Web search for CRAG
- `google-generativeai` - Google Gemini API
- `numpy` - Numerical computing
- `pandas` - Data analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for language model capabilities
- LangChain community for RAG frameworks
- Streamlit team for the excellent web framework
- HuggingFace for embedding models
- FAISS team for vector similarity search

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages carefully
3. Ensure all requirements are installed
4. Verify your API key is set correctly

For additional support, please create an issue with:
- Error message (if any)
- Steps to reproduce
- Your Python version
- Operating system

---

**Happy chatting with your RAG systems!** 🤖✨
