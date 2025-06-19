# 🚀 Multi-RAG Chatbot - Complete Setup Guide

Welcome to the Multi-RAG Chatbot! This guide will help you set up and run the chatbot application that lets you compare different RAG (Retrieval-Augmented Generation) techniques with your documents.

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Chatbot](#running-the-chatbot)
5. [Usage Guide](#usage-guide)
6. [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements
- **Python 3.8 or higher**
- **4GB+ RAM** (recommended for processing documents)
- **2GB+ disk space** for dependencies
- **Internet connection** for downloading models and API access

### Get Your Google API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key (keep it secure!)

## ⚡ Quick Installation

### Option 1: Automated Setup (Recommended)

#### For Windows:
```powershell
# Download and run the PowerShell setup script
.\launch_chatbot.ps1
```

#### For Linux/Mac:
```bash
# Download and run the Python setup script
python launch_chatbot.py
```

### Option 2: Manual Installation

#### Step 1: Clone or Download Files
```bash
# If using git
git clone <repository-url>
cd rag-chatbot

# Or download and extract the ZIP file
```

#### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements_chatbot.txt
```

#### Step 4: Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your Google API key
# GOOGLE_API_KEY=your_actual_api_key_here
```

## 🔑 Configuration

### Setting Up Your API Key

1. **Create .env file:**
   ```bash
   # Create a new file called .env in the project root
   touch .env  # Linux/Mac
   # Or create manually in Windows
   ```

2. **Add your API key:**
   ```env
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

3. **Verify setup:**
   ```python
   # Test your configuration
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key loaded:', bool(os.getenv('GOOGLE_API_KEY')))"
   ```

## 🚀 Running the Chatbot

### Method 1: Using Launcher Scripts

#### Windows PowerShell:
```powershell
.\launch_chatbot.ps1
```

#### Python Launcher:
```bash
python launch_chatbot.py
```

### Method 2: Direct Streamlit Command
```bash
streamlit run chatbot_app.py
```

### Method 3: Advanced Version with Comparisons
```bash
streamlit run advanced_chatbot_app.py
```

### Method 4: Command Line Demo
```bash
python demo.py
```

## 📖 Usage Guide

### Basic Chatbot (chatbot_app.py)

1. **Upload Documents:**
   - Click "Browse files" in the sidebar
   - Select PDF, TXT, CSV, JSON, DOCX, or XLSX files
   - Wait for upload confirmation

2. **Choose RAG Technique:**
   - Select from dropdown: Adaptive RAG, CRAG, Document Augmentation, Basic RAG
   - Read the technique description

3. **Ask Questions:**
   - Type your question in the chat input
   - Press Enter to get a response
   - View the answer with technique information

4. **View History:**
   - See all previous messages
   - Track which technique was used
   - Export chat history as JSON

### Advanced Chatbot (advanced_chatbot_app.py)

1. **Choose Mode:**
   - **Single Technique:** Use one RAG method at a time
   - **Compare All:** Test all techniques simultaneously

2. **Comparison Features:**
   - Side-by-side response comparison
   - Response time analysis
   - Performance metrics and charts

3. **Analytics Dashboard:**
   - Usage statistics
   - Response time comparisons
   - Technique performance analysis

### Demo Script (demo.py)

1. **Comparison Demo:**
   - Tests all RAG techniques with predefined questions
   - Shows performance comparisons
   - Uses sample documents

2. **Interactive Demo:**
   - Choose one RAG technique
   - Ask your own questions
   - Command-line interface

## 📁 Supported File Formats

| Format | Extension | Description | Best For |
|--------|-----------|-------------|----------|
| PDF | .pdf | Portable documents | Research papers, manuals |
| Text | .txt | Plain text | Simple documents, notes |
| CSV | .csv | Comma-separated values | Data tables, spreadsheets |
| JSON | .json | Structured data | Configuration files, APIs |
| Word | .docx | Microsoft Word | Business documents |
| Excel | .xlsx | Microsoft Excel | Data analysis, reports |

## 🔧 RAG Techniques Explained

### 1. Adaptive RAG 🧠
- **How it works:** Automatically classifies your question type (Factual, Analytical, Opinion, Contextual) and chooses the best retrieval strategy
- **Best for:** Mixed question types, general-purpose usage
- **Strengths:** Intelligent adaptation, handles diverse queries

### 2. CRAG (Corrective RAG) 🔍
- **How it works:** Evaluates retrieved document quality; falls back to web search if documents are insufficient
- **Best for:** Situations where document coverage might be incomplete
- **Strengths:** Self-correcting, web search fallback

### 3. Document Augmentation 📝
- **How it works:** Generates synthetic questions from documents to improve retrieval precision
- **Best for:** Improving search accuracy and finding specific information
- **Strengths:** Enhanced document understanding, better matching

### 4. Basic RAG ⚡
- **How it works:** Standard similarity-based retrieval using vector embeddings
- **Best for:** Simple, straightforward question-answering
- **Strengths:** Fast, reliable, easy to understand

## 🎯 Use Cases and Examples

### Business Documents
```
Upload: Company reports, policies, procedures
Questions: "What is our remote work policy?" "Show me Q3 sales figures"
Best RAG: Adaptive RAG or Document Augmentation
```

### Research Papers
```
Upload: Academic papers, research documents
Questions: "What methodology was used?" "What are the key findings?"
Best RAG: CRAG or Adaptive RAG
```

### Technical Documentation
```
Upload: API docs, user manuals, guides
Questions: "How do I configure this feature?" "What are the requirements?"
Best RAG: Document Augmentation or Basic RAG
```

### Data Analysis
```
Upload: CSV files, Excel reports
Questions: "What are the top products?" "Show me sales trends"
Best RAG: Adaptive RAG (handles analytical queries well)
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### "GOOGLE_API_KEY not found"
```bash
# Solution 1: Check .env file exists and contains your key
cat .env  # Linux/Mac
type .env  # Windows

# Solution 2: Set environment variable directly
export GOOGLE_API_KEY="your_key_here"  # Linux/Mac
set GOOGLE_API_KEY=your_key_here  # Windows CMD
```

#### "No module named 'streamlit'"
```bash
# Solution: Install requirements
pip install -r requirements_chatbot.txt

# If still failing, try upgrading pip
python -m pip install --upgrade pip
pip install streamlit
```

#### "Error loading documents"
```bash
# Check file format is supported
# Try with smaller files first
# Ensure files are not corrupted

# For PDF issues, try alternative:
pip install pdfplumber pypdf2
```

#### "Rate limit exceeded"
```bash
# Solution: Wait between requests
# Check Google API quotas at https://console.cloud.google.com/
# Consider upgrading your API plan
```

#### "Port 8501 already in use"
```bash
# Solution: Use different port
streamlit run chatbot_app.py --server.port 8502

# Or kill existing process
# Windows: taskkill /f /im python.exe
# Linux/Mac: pkill -f streamlit
```

#### Memory/Performance Issues
```bash
# Solutions:
# 1. Use smaller documents
# 2. Increase system RAM
# 3. Close other applications
# 4. Try Basic RAG for faster processing
```

### Debug Mode
```bash
# Run with debug information
streamlit run chatbot_app.py --logger.level debug

# Check system resources
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

## 🔧 Advanced Configuration

### Custom Settings
Create a `config.py` file for custom settings:
```python
# config.py
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 4000
TEMPERATURE = 0
SIMILARITY_TOP_K = 5
```

### Model Configuration
Modify model settings in the RAG classes:
```python
# For different Gemini models
model = "gemini-1.5-pro"  # More capable but slower
model = "gemini-1.5-flash"  # Faster, default
```

### Adding Custom Documents
```bash
# Add documents to data/ directory
mkdir -p data
cp your_documents.pdf data/
```

## 📊 Performance Optimization

### For Better Speed:
1. Use **Basic RAG** for simple queries
2. Reduce `chunk_size` in configuration
3. Use fewer documents at once
4. Close unnecessary browser tabs

### For Better Accuracy:
1. Use **Document Augmentation** for specific searches
2. Use **Adaptive RAG** for mixed queries
3. Upload relevant documents only
4. Use clear, specific questions

### For Handling Large Documents:
1. Split large files into smaller chunks
2. Use **CRAG** for comprehensive coverage
3. Increase system memory if possible
4. Process documents one at a time

## 🤝 Getting Help

### Self-Help Resources:
1. Check this troubleshooting guide
2. Review error messages carefully
3. Test with sample documents first
4. Try different RAG techniques

### Error Reporting:
When reporting issues, include:
- Error message (full text)
- Python version: `python --version`
- Operating system
- Steps to reproduce
- File types being used

### Community:
- Check GitHub issues
- Review documentation
- Test with provided demo files

## 🎉 Success Checklist

✅ Python 3.8+ installed  
✅ Google API key obtained  
✅ Dependencies installed  
✅ .env file configured  
✅ Sample documents available  
✅ Chatbot launches successfully  
✅ Can upload and process documents  
✅ Can ask questions and get responses  
✅ Can compare different RAG techniques  

## 🚀 Next Steps

Once you have the chatbot running:

1. **Experiment with different document types**
2. **Try various RAG techniques to see which works best for your use case**
3. **Use the comparison mode to understand technique differences**
4. **Export and analyze your chat history**
5. **Customize the interface for your specific needs**

---

**Congratulations! You're now ready to explore the power of different RAG techniques with your own documents!** 🎉

For additional features and updates, check the project repository regularly.
