# Multi-RAG Chatbot v2

A sophisticated multi-document RAG (Retrieval-Augmented Generation) chatbot with CRAG (Corrective RAG), source highlighting, web search capabilities, and a modern Streamlit interface.

## Features

- **Multi-Document Support**: Load and process multiple document types (PDF, DOCX, TXT, MD)
- **CRAG (Corrective RAG)**: Intelligent retrieval evaluation and correction
- **Source Highlighting**: Visual highlighting of relevant source passages
- **Web Search Integration**: Fallback to web search when document retrieval is insufficient
- **API Key Management**: Automatic rotation and management of multiple API keys
- **Modern UI**: Clean, responsive Streamlit interface
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Comprehensive Testing**: Unit and integration tests included

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multi-rag-chatbot.git
   cd multi-rag-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Development Installation

For development with testing and linting tools:

```bash
pip install -e ".[dev]"
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Gemini API Keys (comma-separated for rotation)
GEMINI_API_KEYS=your_api_key_1,your_api_key_2,your_api_key_3

# Google Search API (optional, for web search)
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_RESULTS=5
MAX_WEB_RESULTS=3
LLM_TEMPERATURE=0.7
LLM_MODEL_NAME=gemini-pro

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Custom Configuration

You can also modify settings directly in `config/settings.py` for more advanced configuration options.

## Usage

### Web Interface

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload documents** using the sidebar file uploader

3. **Ask questions** in the chat interface

4. **View sources** and highlighted passages in the response

### Command Line Interface

The application also provides a CLI for batch processing and automation:

```bash
# Interactive mode
python cli.py --documents ./data/sample_documents --interactive

# Single query mode
python cli.py --documents ./data/sample_documents --query "What is machine learning?"

# Web search only mode
python cli.py --query "Latest developments in AI"
```

### API Usage

You can also use the components programmatically:

```python
from src.core.crag import CRAGProcessor
from src.core.retrieval import DocumentRetriever
from src.core.evaluation import RetrievalEvaluator
from src.llm.gemini import GeminiLLM
from src.search.web import DuckDuckGoSearcher

# Create components
llm = GeminiLLM(api_key="your_api_key")
retriever = DocumentRetriever(vectorstore=your_vectorstore)
evaluator = RetrievalEvaluator(llm=llm)
web_searcher = DuckDuckGoSearcher()

# Create CRAG processor
crag = CRAGProcessor(
    retriever=retriever,
    llm=llm,
    evaluator=evaluator,
    web_search=web_searcher
)

# Process queries
result = crag.process_query("Your question here")
print(result["response"])
```

## Project Structure

```
multi-rag-chatbot/
├── app.py                    # Main Streamlit application
├── cli.py                    # Command-line interface
├── setup.py                  # Package setup
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore file
├── README.md                # This file
├── config/                  # Configuration files
│   ├── __init__.py
│   ├── settings.py          # Application settings
│   └── prompts.py           # LLM prompts
├── src/                     # Source code
│   ├── core/                # Core RAG functionality
│   │   ├── __init__.py
│   │   ├── crag.py          # CRAG processor
│   │   ├── retrieval.py     # Document retrieval
│   │   ├── evaluation.py    # Retrieval evaluation
│   │   └── knowledge.py     # Knowledge refinement
│   ├── document/            # Document processing
│   │   ├── __init__.py
│   │   ├── loader.py        # Document loaders
│   │   ├── chunking.py      # Text chunking
│   │   └── viewer.py        # Document viewing
│   ├── llm/                 # Language model integration
│   │   ├── __init__.py
│   │   ├── gemini.py        # Gemini LLM
│   │   └── api_manager.py   # API key management
│   ├── search/              # Web search functionality
│   │   ├── __init__.py
│   │   └── web.py           # Web searchers
│   ├── ui/                  # User interface
│   │   ├── __init__.py
│   │   ├── streamlit_app.py # Main Streamlit app
│   │   ├── components.py    # UI components
│   │   └── styles.py        # CSS styles
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── logging.py       # Logging setup
│       ├── exceptions.py    # Custom exceptions
│       └── helpers.py       # Helper functions
├── tests/                   # Test files
│   ├── conftest.py          # Test configuration
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test fixtures
├── data/                    # Data directory
│   └── sample_documents/    # Sample documents
├── logs/                    # Log files
├── docs/                    # Documentation
└── scripts/                 # Utility scripts
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_crag.py

# Run integration tests
pytest tests/integration/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write tests for new features
- Update documentation as needed
- Use type hints where appropriate

### Code Quality

The project uses several tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your Gemini API keys are correctly set in the `.env` file
   - Check that your API keys have sufficient quota

2. **Document Loading Issues**
   - Verify that your documents are in supported formats (PDF, DOCX, TXT, MD)
   - Check file permissions and paths

3. **Memory Issues**
   - Reduce chunk size in configuration for large documents
   - Consider using fewer documents or smaller files

4. **Web Search Issues**
   - Check internet connectivity
   - Verify Google Search API credentials if using Google Search

### Logging

Enable verbose logging to debug issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use the CLI verbose flag:

```bash
python cli.py --verbose --query "test query"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google for the Gemini API
- Streamlit for the amazing web app framework
- The open-source community for the various libraries used in this project

## Changelog

### v2.0.0 (Current)

- Complete refactor with modular architecture
- Added CRAG (Corrective RAG) functionality
- Improved source highlighting
- Enhanced web search integration
- Added comprehensive testing
- CLI interface
- API key management
- Better error handling and logging

### v1.0.0

- Initial release
- Basic RAG functionality
- Simple document upload
- Basic Streamlit interface
