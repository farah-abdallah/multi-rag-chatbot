"""
Command-line interface for the RAG Chatbot.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.core.crag import CRAGProcessor
from src.core.retrieval import DocumentRetriever
from src.core.evaluation import RetrievalEvaluator
from src.document.loader import load_documents
from src.document.chunking import TextChunker
from src.llm.gemini import GeminiLLM
from src.llm.api_manager import APIKeyManager
from src.search.web import DuckDuckGoSearcher
from src.utils.logging import setup_logging
from config.settings import Settings


def create_rag_processor(settings: Settings, documents_path: str = None):
    """Create a RAG processor with the given settings."""
    
    # Load documents if path provided
    documents = []
    if documents_path and os.path.exists(documents_path):
        if os.path.isdir(documents_path):
            # Load all files in directory
            file_paths = []
            for ext in ['.txt', '.pdf', '.docx', '.md']:
                file_paths.extend(Path(documents_path).glob(f"*{ext}"))
            documents = load_documents([str(p) for p in file_paths])
        else:
            # Single file
            documents = load_documents([documents_path])
    
    # Chunk documents
    if documents:
        chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        chunked_docs = chunker.chunk_documents(documents)
        print(f"Loaded and chunked {len(documents)} documents into {len(chunked_docs)} chunks")
    else:
        print("No documents loaded. Web search only mode.")
        chunked_docs = []
    
    # Create components
    api_keys = settings.GEMINI_API_KEYS
    if not api_keys:
        raise ValueError("No Gemini API keys found. Please set GEMINI_API_KEYS in your environment.")
    
    api_manager = APIKeyManager(api_keys)
    llm = GeminiLLM(
        api_key=api_manager.get_current_key(),
        model_name=settings.LLM_MODEL_NAME
    )
    
    # Create retriever (mock for CLI, would need proper vector store)
    retriever = DocumentRetriever(vectorstore=None, k=settings.MAX_RETRIEVAL_RESULTS)
    
    # Create evaluator
    evaluator = RetrievalEvaluator(llm=llm)
    
    # Create web searcher
    web_searcher = DuckDuckGoSearcher(max_results=settings.MAX_WEB_RESULTS)
    
    # Create CRAG processor
    crag_processor = CRAGProcessor(
        retriever=retriever,
        llm=llm,
        evaluator=evaluator,
        web_search=web_searcher
    )
    
    return crag_processor


def interactive_mode(crag_processor: CRAGProcessor):
    """Run the chatbot in interactive mode."""
    print("RAG Chatbot - Interactive Mode")
    print("Type 'quit' or 'exit' to stop, 'help' for commands")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help - Show this help message")
                print("  quit/exit/q - Exit the chatbot")
                print("  Any other input - Ask a question")
                continue
            
            if not user_input:
                continue
            
            print("\nProcessing your question...")
            
            # Process the query
            result = crag_processor.process_query(user_input)
            
            print(f"\nBot: {result['response']}")
            
            # Show sources if available
            if result.get('sources'):
                print(f"\nSources ({len(result['sources'])} found):")
                for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                    source_name = source['metadata'].get('source', 'Unknown')
                    print(f"  {i}. {source_name}")
            
            # Show evaluation
            if result.get('evaluation'):
                eval_data = result['evaluation']
                print(f"\nEvaluation: {eval_data.get('quality', 'N/A')} "
                      f"(confidence: {eval_data.get('confidence', 0):.2f})")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def single_query_mode(crag_processor: CRAGProcessor, query: str):
    """Process a single query and return the result."""
    print(f"Processing query: {query}")
    print("-" * 50)
    
    try:
        result = crag_processor.process_query(query)
        
        print(f"Answer: {result['response']}")
        
        if result.get('sources'):
            print(f"\nSources ({len(result['sources'])} found):")
            for i, source in enumerate(result['sources'], 1):
                source_name = source['metadata'].get('source', 'Unknown')
                print(f"  {i}. {source_name}")
        
        if result.get('evaluation'):
            eval_data = result['evaluation']
            print(f"\nEvaluation: {eval_data.get('quality', 'N/A')} "
                  f"(confidence: {eval_data.get('confidence', 0):.2f})")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument(
        "--documents", "-d",
        help="Path to documents directory or single document file"
    )
    parser.add_argument(
        "--query", "-q",
        help="Single query to process (non-interactive mode)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (default if no query provided)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Load settings
    settings = Settings()
    
    try:
        # Create RAG processor
        crag_processor = create_rag_processor(settings, args.documents)
        
        # Run in appropriate mode
        if args.query:
            # Single query mode
            exit_code = single_query_mode(crag_processor, args.query)
            sys.exit(exit_code)
        else:
            # Interactive mode
            interactive_mode(crag_processor)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
