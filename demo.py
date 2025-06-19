#!/usr/bin/env python3
"""
Demo Script for Multi-RAG Chatbot

This script demonstrates the different RAG techniques with sample questions
and provides performance comparisons.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Set up logging for backend visibility
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to path to import our modules
sys.path.append(str(Path(__file__).parent))

from adaptive_rag import AdaptiveRAG
from crag import CRAG
from document_augmentation import DocumentProcessor, SentenceTransformerEmbeddings, load_document_content
from helper_functions import encode_document
from explainable_retrieval import ExplainableRAGMethod

def print_header():
    """Print demo header"""
    print("=" * 80)
    print("🤖 MULTI-RAG CHATBOT DEMO")
    print("=" * 80)
    print("This demo showcases 5 different RAG techniques with sample documents:")
    print("  • Adaptive RAG - Dynamically adapts strategy based on query type")
    print("  • CRAG - Corrective RAG with relevance evaluation") 
    print("  • Document Augmentation - Enhanced documents with generated questions")
    print("  • Basic RAG - Standard similarity-based retrieval")
    print("  • Explainable Retrieval - Provides explanations for retrieved content")
    print()
    print("💡 Backend logging is enabled - watch for detailed processing info!")
    print()

def check_sample_data():
    """Check if sample data files exist"""
    sample_files = [
        "data/sample_text.txt",
        "data/sample_products.csv", 
        "data/sample_company.json"
    ]
    
    existing_files = []
    for file_path in sample_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"⚠️ Sample file not found: {file_path}")
    
    if not existing_files:
        print("❌ No sample files found. Please ensure the data/ directory contains sample files.")
        return None
    
    print(f"✅ Found {len(existing_files)} sample files:")
    for file_path in existing_files:
        print(f"   📄 {file_path}")
    
    return existing_files[0]  # Return the first available file

def test_rag_technique(technique_name, rag_system, query):
    """Test a single RAG technique"""
    print(f"\n{'─' * 60}")
    print(f"🔧 Testing: {technique_name}")
    print(f"{'─' * 60}")
    print(f"📝 Query: {query}")
    print("🔄 Processing...")
    
    # Enable more detailed logging for Explainable Retrieval
    if technique_name == "Explainable Retrieval":
        print("🔍 Watch the console for detailed backend processing logs...")
    
    start_time = time.time()
    
    try:
        if technique_name == "Adaptive RAG":
            response = rag_system.answer(query)
        elif technique_name == "CRAG":
            response = rag_system.run(query)
        elif technique_name == "Document Augmentation":
            docs = rag_system.get_relevant_documents(query)
            if docs:
                from document_augmentation import generate_answer
                context = docs[0].metadata.get('text', docs[0].page_content)
                response = generate_answer(context, query)
            else:
                response = "No relevant documents found."
        elif technique_name == "Basic RAG":
            docs = rag_system.similarity_search(query, k=3)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                response = f"Based on the documents: {context[:300]}..."
            else:
                response = "No relevant documents found."
        elif technique_name == "Explainable Retrieval":
            # Use the answer method for comprehensive response
            response = rag_system.answer(query)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"✅ Response ({response_time:.2f}s):")
        print(f"💬 {response[:500]}{'...' if len(response) > 500 else ''}")
        
        # Show additional info for Explainable Retrieval
        if technique_name == "Explainable Retrieval":
            print("\n🔍 For detailed explanations, check the backend logs above!")
        
        return response, response_time
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        print(f"❌ Error ({response_time:.2f}s): {str(e)}")
        logger.error(f"Error in {technique_name}: {str(e)}")
        return None, response_time

def run_comparison_demo(sample_file):
    """Run a comparison demo with all RAG techniques"""
    print(f"\n🚀 COMPARISON DEMO")
    print(f"Using sample file: {sample_file}")
    print("📊 Backend logging is enabled - watch for detailed processing!")
    
    # Sample queries for testing
    queries = [
        "What is artificial intelligence?",
        "What are the main applications mentioned?",
        "Explain the key concepts discussed.",
        "What are the future trends?"
    ]
    
    # Initialize RAG systems
    print("\n🔧 Initializing RAG systems...")
    logger.info("Starting RAG systems initialization")
    
    rag_systems = {}
    
    try:
        print("   Loading Adaptive RAG...")
        logger.info("Initializing Adaptive RAG")
        rag_systems["Adaptive RAG"] = AdaptiveRAG(file_paths=[sample_file])
        print("   ✅ Adaptive RAG loaded")
        logger.info("Adaptive RAG loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading Adaptive RAG: {e}")
        logger.error(f"Error loading Adaptive RAG: {e}")
    
    try:
        print("   Loading CRAG...")
        logger.info("Initializing CRAG")
        rag_systems["CRAG"] = CRAG(sample_file)
        print("   ✅ CRAG loaded")
        logger.info("CRAG loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading CRAG: {e}")
        logger.error(f"Error loading CRAG: {e}")
    
    try:
        print("   Loading Document Augmentation...")
        logger.info("Initializing Document Augmentation")
        content = load_document_content(sample_file)
        embedding_model = SentenceTransformerEmbeddings()
        processor = DocumentProcessor(content, embedding_model, sample_file)
        rag_systems["Document Augmentation"] = processor.run()
        print("   ✅ Document Augmentation loaded")
        logger.info("Document Augmentation loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading Document Augmentation: {e}")
        logger.error(f"Error loading Document Augmentation: {e}")
    
    try:
        print("   Loading Basic RAG...")
        logger.info("Initializing Basic RAG")
        rag_systems["Basic RAG"] = encode_document(sample_file)
        print("   ✅ Basic RAG loaded")
        logger.info("Basic RAG loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading Basic RAG: {e}")
        logger.error(f"Error loading Basic RAG: {e}")
    
    try:
        print("   Loading Explainable Retrieval...")
        logger.info("Initializing Explainable Retrieval")
        content = load_document_content(sample_file)
        # Split content into chunks for Explainable Retrieval
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(content)
        doc_name = os.path.basename(sample_file)
        texts_with_source = [f"[Source: {doc_name}] {chunk}" for chunk in chunks]
        rag_systems["Explainable Retrieval"] = ExplainableRAGMethod(texts_with_source)
        print("   ✅ Explainable Retrieval loaded")
        logger.info("Explainable Retrieval loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading Explainable Retrieval: {e}")
        logger.error(f"Error loading Explainable Retrieval: {e}")
    
    if not rag_systems:
        print("❌ No RAG systems could be loaded. Please check your setup.")
        logger.error("No RAG systems could be loaded")
        return
    
    print(f"\n✅ Successfully loaded {len(rag_systems)} RAG systems")
    logger.info(f"Successfully loaded {len(rag_systems)} RAG systems: {list(rag_systems.keys())}")
    
    # Test each query with all available systems
    for i, query in enumerate(queries, 1):
        print(f"\n{'═' * 80}")
        print(f"🔍 QUERY {i}: {query}")
        print(f"{'═' * 80}")
        
        results = {}
        
        for technique_name, rag_system in rag_systems.items():
            response, response_time = test_rag_technique(technique_name, rag_system, query)
            if response:
                results[technique_name] = (response, response_time)
        
        # Show comparison summary
        if len(results) > 1:
            print(f"\n📊 COMPARISON SUMMARY:")
            print(f"{'Technique':<25} {'Time (s)':<10} {'Response Length':<15}")
            print("─" * 50)
            
            for technique, (response, response_time) in results.items():
                print(f"{technique:<25} {response_time:<10.2f} {len(response):<15}")
        
        if i < len(queries):
            input("\n⏸️ Press Enter to continue to next query...")

def interactive_demo(sample_file):
    """Run interactive demo where user can ask questions"""
    print(f"\n🎯 INTERACTIVE DEMO")
    print("Ask your own questions! Type 'quit' to exit.")
    print("💡 Backend logging is enabled - watch for detailed processing!")
    
    # Let user choose a RAG technique
    techniques = ["Adaptive RAG", "CRAG", "Document Augmentation", "Basic RAG", "Explainable Retrieval"]
    
    print("\nAvailable RAG techniques:")
    for i, technique in enumerate(techniques, 1):
        print(f"   {i}. {technique}")
    
    while True:
        try:
            choice = input("\nChoose a technique (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                selected_technique = techniques[int(choice) - 1]
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        except (ValueError, KeyboardInterrupt):
            print("\nDemo cancelled.")
            return
    
    # Load the selected RAG system
    print(f"\n🔧 Loading {selected_technique}...")
    logger.info(f"Loading {selected_technique} for interactive demo")
    
    try:
        if selected_technique == "Adaptive RAG":
            rag_system = AdaptiveRAG(file_paths=[sample_file])
        elif selected_technique == "CRAG":
            rag_system = CRAG(sample_file)
        elif selected_technique == "Document Augmentation":
            content = load_document_content(sample_file)
            embedding_model = SentenceTransformerEmbeddings()
            processor = DocumentProcessor(content, embedding_model, sample_file)
            rag_system = processor.run()
        elif selected_technique == "Basic RAG":
            rag_system = encode_document(sample_file)
        elif selected_technique == "Explainable Retrieval":
            content = load_document_content(sample_file)
            # Split content into chunks for Explainable Retrieval
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(content)
            doc_name = os.path.basename(sample_file)
            texts_with_source = [f"[Source: {doc_name}] {chunk}" for chunk in chunks]
            rag_system = ExplainableRAGMethod(texts_with_source)
        
        print(f"✅ {selected_technique} loaded successfully!")
        logger.info(f"{selected_technique} loaded successfully for interactive demo")
        
    except Exception as e:
        print(f"❌ Error loading {selected_technique}: {e}")
        logger.error(f"Error loading {selected_technique}: {e}")
        return
    
    # Interactive query loop
    print(f"\n💬 Chat with {selected_technique}")
    if selected_technique == "Explainable Retrieval":
        print("🔍 Special note: Watch the console for detailed explanation logs!")
    print("Type your questions below (type 'quit' to exit):")
    
    while True:
        try:
            query = input("\n🙋 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for trying the demo!")
                break
            
            if not query:
                continue
            
            response, response_time = test_rag_technique(selected_technique, rag_system, query)
            
        except KeyboardInterrupt:
            print("\n👋 Demo ended by user.")
            break

def adjust_logging_level():
    """Allow user to adjust logging verbosity"""
    print(f"\n⚙️ LOGGING CONTROL")
    print("Current logging levels:")
    print("1. 🔇 CRITICAL - Only critical errors")
    print("2. ❌ ERROR - Errors and critical")
    print("3. ⚠️ WARNING - Warnings, errors, and critical")
    print("4. ℹ️ INFO - Info, warnings, errors, and critical (current)")
    print("5. 🔍 DEBUG - All messages including debug info")
    
    while True:
        try:
            choice = input("\nChoose logging level (1-5): ").strip()
            
            if choice == '1':
                logging.getLogger().setLevel(logging.CRITICAL)
                print("✅ Logging set to CRITICAL level")
                break
            elif choice == '2':
                logging.getLogger().setLevel(logging.ERROR)
                print("✅ Logging set to ERROR level")
                break
            elif choice == '3':
                logging.getLogger().setLevel(logging.WARNING)
                print("✅ Logging set to WARNING level")
                break
            elif choice == '4':
                logging.getLogger().setLevel(logging.INFO)
                print("✅ Logging set to INFO level")
                break
            elif choice == '5':
                logging.getLogger().setLevel(logging.DEBUG)
                print("✅ Logging set to DEBUG level")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
                
        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            break

def explainable_focus_demo(sample_file):
    """Deep dive demo focused on Explainable Retrieval with detailed logging"""
    print(f"\n🔍 EXPLAINABLE RETRIEVAL FOCUS DEMO")
    print("This demo focuses specifically on Explainable Retrieval with maximum logging visibility.")
    print("💡 All backend operations will be logged in detail!")
    
    # Temporarily increase logging level for maximum visibility
    original_level = logging.getLogger().getLevel()
    logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print(f"\n🔧 Loading Explainable Retrieval...")
        logger.info("Starting Explainable Retrieval Focus Demo")
        
        content = load_document_content(sample_file)
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(content)
        doc_name = os.path.basename(sample_file)
        texts_with_source = [f"[Source: {doc_name}] {chunk}" for chunk in chunks]
        rag_system = ExplainableRAGMethod(texts_with_source)
        
        print(f"✅ Explainable Retrieval loaded successfully!")
        
        # Sample queries specifically for testing explanations
        test_queries = [
            "What is the main topic discussed?",
            "What are the key points mentioned?",
            "Can you explain the central concepts?",
            "What conclusions can be drawn?"
        ]
        
        print(f"\n📊 Testing with {len(test_queries)} sample queries...")
        print("🔍 Watch the detailed logs below to see how explanations are generated!")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'═' * 60}")
            print(f"🔍 TEST QUERY {i}: {query}")
            print(f"{'═' * 60}")
            
            response, response_time = test_rag_technique("Explainable Retrieval", rag_system, query)
            
            if response:
                print(f"\n📈 Performance: {response_time:.2f} seconds")
                print(f"📏 Response length: {len(response)} characters")
            
            if i < len(test_queries):
                input(f"\n⏸️ Press Enter to continue to query {i+1}...")
        
        # Interactive session
        print(f"\n💬 INTERACTIVE SESSION")
        print("Now you can ask your own questions. Type 'quit' to exit.")
        
        while True:
            try:
                query = input("\n🙋 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                response, response_time = test_rag_technique("Explainable Retrieval", rag_system, query)
                
            except KeyboardInterrupt:
                print("\nExiting focus demo...")
                break
        
    except Exception as e:
        print(f"❌ Error in focus demo: {e}")
        logger.error(f"Error in explainable focus demo: {e}")
    
    finally:
        # Restore original logging level
        logging.getLogger().setLevel(original_level)
        print(f"\n✅ Focus demo completed. Logging level restored.")

def main():
    """Main demo function"""
    print_header()
    
    # Check for sample data
    sample_file = check_sample_data()
    if not sample_file:
        print("\n💡 To run the demo, please:")
        print("   1. Ensure you have sample files in the data/ directory")
        print("   2. Or run the full chatbot app: streamlit run chatbot_app.py")
        return
    
    print(f"\n🎯 DEMO OPTIONS:")
    print("1. 🔄 Comparison Demo - Test all 5 RAG techniques with predefined questions")
    print("2. 🎯 Interactive Demo - Ask your own questions with one RAG technique")
    print("3. � Explainable Retrieval Focus - Deep dive into explainable retrieval with logging")
    print("4. �🚀 Launch Full Chatbot - Open the complete web interface")
    print("5. ⚙️ Adjust Logging Level - Control backend logging verbosity")
    print("6. ❌ Exit")
    
    while True:
        try:
            choice = input("\nChoose an option (1-6): ").strip()
            
            if choice == '1':
                run_comparison_demo(sample_file)
                break
            elif choice == '2':
                interactive_demo(sample_file)
                break
            elif choice == '3':
                explainable_focus_demo(sample_file)
                break
            elif choice == '4':
                print("\n🚀 Launching full chatbot...")
                print("Run: streamlit run chatbot_app.py")
                print("Or: python launch_chatbot.py")
                break
            elif choice == '5':
                adjust_logging_level()
                continue  # Return to menu after adjusting logging
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
                
        except KeyboardInterrupt:
            print("\n👋 Demo cancelled.")
            break

if __name__ == "__main__":
    main()
