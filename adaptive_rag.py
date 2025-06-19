"""
Adaptive RAG System using Google Gemini API

To install required dependencies:
pip install langchain langchain-google-genai langchain-community faiss-cpu python-dotenv

Make sure to set GOOGLE_API_KEY in your environment or .env file.

Current models used:
- Chat model: gemini-1.5-flash (faster, cost-effective)
- Embedding model: models/text-embedding-004

Alternative models available:
- Chat: gemini-1.5-pro (more capable but slower/expensive)
"""

import os
import sys
import glob
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

from langchain_core.retrievers import BaseRetriever
from typing import List, Dict, Any, Union
from langchain.docstore.document import Document
import google.generativeai as genai

# Multi-format document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, JSONLoader,
    UnstructuredWordDocumentLoader, UnstructuredExcelLoader
)
import glob

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path since we work with notebooks
#from helper_functions import *
#from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the Google API key environment variable
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)


class SimpleGeminiLLM:
    """Simple wrapper for Google Gemini API"""
    
    def __init__(self, model="gemini-1.5-flash", max_tokens=4000, temperature=0):
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = genai.GenerativeModel(model)
    
    def invoke(self, prompt_text):
        """Invoke the model with a text prompt"""
        try:
            response = self.model.generate_content(
                prompt_text,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            )
            return SimpleResponse(response.text)
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


class SimpleResponse:
    """Simple response wrapper"""
    def __init__(self, content):
        self.content = content


def load_documents_from_files(file_paths: Union[str, List[str]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Load documents from various file formats and return as text chunks.
    Supports: PDF, TXT, CSV, JSON, DOCX, XLSX
    """
    if isinstance(file_paths, str):
        # Handle single file or directory
        if os.path.isdir(file_paths):
            # Load all supported files from directory
            file_paths = []
            for ext in ['*.pdf', '*.txt', '*.csv', '*.json', '*.docx', '*.xlsx']:
                file_paths.extend(glob.glob(os.path.join(file_paths, ext)))
        else:
            file_paths = [file_paths]
    
    all_documents = []  # Store Document objects for splitting
    
    for file_path in file_paths:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            print(f"Loading {file_extension.upper()} file: {file_path}")
            
            # Choose appropriate loader based on file type
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
            elif file_extension == '.json':
                loader = JSONLoader(file_path, jq_schema='.', text_content=False)
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
            else:
                print(f"Warning: Unsupported file type {file_extension}, skipping {file_path}")
                continue
            
            # Load documents
            documents = loader.load()
            all_documents.extend(documents)
            
            print(f"✅ Successfully loaded {len(documents)} pages from {file_path}")
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
            continue
    
    # Now split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    
    print(f"� Total pages loaded: {len(all_documents)}")
    print(f"🔄 Splitting into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    
    # Split documents into smaller chunks
    split_docs = text_splitter.split_documents(all_documents)
    
    # Extract text content from split documents
    all_texts = [doc.page_content for doc in split_docs]
    
    print(f"📚 Total text chunks created: {len(all_texts)}")
    
    # Show chunk size distribution for debugging
    chunk_sizes = [len(text) for text in all_texts]
    if chunk_sizes:
        print(f"📊 Chunk sizes - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)//len(chunk_sizes)}")
    
    return all_texts


def load_documents_from_directory(directory_path: str) -> List[str]:
    """
    Load all supported documents from a directory.
    """
    return load_documents_from_files(directory_path)


# Define all the required classes and strategies
# Removed Pydantic models - using direct text parsing instead

class QueryClassifier:
    def __init__(self):
        self.llm = SimpleGeminiLLM(model="gemini-1.5-flash", temperature=0, max_tokens=4000)

    def classify(self, query):
        print("Classifying query...")
        prompt_text = f"""Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.
Respond with ONLY one word: Factual, Analytical, Opinion, or Contextual.

Query: {query}
Category:"""
        
        result = self.llm.invoke(prompt_text)
        
        # Extract the category from the response
        category = result.content.strip()
        
        # Ensure it's a valid category
        valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
        for valid_cat in valid_categories:
            if valid_cat.lower() in category.lower():
                return valid_cat
        
        # Default to Factual if classification fails
        return "Factual"


class BaseRetrievalStrategy:
    def __init__(self, texts):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        self.documents = text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        self.llm = SimpleGeminiLLM(model="gemini-1.5-flash", temperature=0, max_tokens=4000)

    def retrieve(self, query, k=4):
        return self.db.similarity_search(query, k=k)


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("Retrieving factual information...")
        enhanced_query_prompt_text = f"Enhance this factual query for better information retrieval: {query}"
        enhanced_query_result = self.llm.invoke(enhanced_query_prompt_text)
        enhanced_query = enhanced_query_result.content
        print(f'Enhanced query: {enhanced_query}')

        docs = self.db.similarity_search(enhanced_query, k=k * 2)

        ranked_docs = []
        print("Ranking documents...")
        for doc in docs:
            ranking_prompt_text = f"On a scale of 1-10, how relevant is this document to the query: '{enhanced_query}'?\nDocument: {doc.page_content}\nRelevance score (just the number):"
            
            result = self.llm.invoke(ranking_prompt_text)
            try:
                score = float(result.content.strip())
            except ValueError:
                # Extract number from response if parsing fails
                import re
                numbers = re.findall(r'\d+\.?\d*', result.content)
                score = float(numbers[0]) if numbers else 5.0
            
            ranked_docs.append((doc, score))

        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("Retrieving analytical information...")
        sub_queries_prompt_text = f"Generate {k} sub-questions for: {query}\nList them as numbered items (1. 2. 3. etc.):"
        
        result = self.llm.invoke(sub_queries_prompt_text)
        
        # Extract sub-queries from the response
        import re
        sub_queries = re.findall(r'\d+\.\s*(.+)', result.content)
        if not sub_queries:
            # Fallback: split by lines and clean
            lines = result.content.strip().split('\n')
            sub_queries = [line.strip() for line in lines if line.strip()][:k]
        
        print(f'Sub-queries: {sub_queries}')

        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k=2))

        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)])
        diversity_prompt_text = f"Select the most diverse and relevant set of {k} documents for the query: '{query}'\nDocuments: {docs_text}\nRespond with only the document numbers separated by commas (e.g., 0, 2, 5):"
        
        result = self.llm.invoke(diversity_prompt_text)
        
        # Extract indices from the response
        import re
        numbers = re.findall(r'\d+', result.content)
        selected_indices = [int(num) for num in numbers[:k]]
        
        if not selected_indices:
            # Fallback: select first k documents
            selected_indices = list(range(min(k, len(all_docs))))

        return [all_docs[i] for i in selected_indices if i < len(all_docs)]


class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=3):
        print("Retrieving opinions...")
        viewpoints_prompt_text = f"Identify {k} distinct viewpoints or perspectives on the topic: {query}"
        viewpoints_result = self.llm.invoke(viewpoints_prompt_text)
        viewpoints = viewpoints_result.content.split('\n')
        print(f'Viewpoints: {viewpoints}')

        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_search(f"{query} {viewpoint}", k=2))

        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(all_docs)])
        opinion_prompt_text = f"Classify these documents into distinct opinions on '{query}' and select the {k} most representative and diverse viewpoints:\nDocuments: {docs_text}\nSelected indices (respond with only numbers separated by commas):"
        
        result = self.llm.invoke(opinion_prompt_text)
        
        # Extract indices from the response
        import re
        numbers = re.findall(r'\d+', result.content)
        selected_indices = [int(num) for num in numbers[:k]]
        
        if not selected_indices:
            # Fallback: select first k documents
            selected_indices = list(range(min(k, len(all_docs))))

        return [all_docs[i] for i in selected_indices if i < len(all_docs)]


class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4, user_context=None):
        print("Retrieving contextual information...")
        context_prompt_text = f"Given the user context: {user_context or 'No specific context provided'}\nReformulate the query to best address the user's needs: {query}"
        contextualized_query_result = self.llm.invoke(context_prompt_text)
        contextualized_query = contextualized_query_result.content
        print(f'Contextualized query: {contextualized_query}')

        docs = self.db.similarity_search(contextualized_query, k=k * 2)

        ranked_docs = []
        for doc in docs:
            ranking_prompt_text = f"Given the query: '{contextualized_query}' and user context: '{user_context or 'No specific context provided'}', rate the relevance of this document on a scale of 1-10:\nDocument: {doc.page_content}\nRelevance score (just the number):"
            
            result = self.llm.invoke(ranking_prompt_text)
            try:
                score = float(result.content.strip())
            except ValueError:
                # Extract number from response if parsing fails
                import re
                numbers = re.findall(r'\d+\.?\d*', result.content)
                score = float(numbers[0]) if numbers else 5.0
            
            ranked_docs.append((doc, score))

        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:k]]


# Define the main Adaptive RAG class
class AdaptiveRAG:
    def __init__(self, texts: List[str] = None, file_paths: Union[str, List[str]] = None):
        """
        Initialize AdaptiveRAG with either text strings or file paths.
        
        Args:
            texts: List of text strings (original functionality)
            file_paths: File path(s) to load documents from (new functionality)
        """
        # Load texts from files if file_paths provided
        if file_paths is not None:
            print("Loading documents from files...")
            loaded_texts = load_documents_from_files(file_paths)
            if texts is not None:
                # Combine provided texts with loaded texts
                texts = texts + loaded_texts
            else:
                texts = loaded_texts
        
        # Ensure we have some texts to work with
        if not texts:
            raise ValueError("No texts provided. Please provide either 'texts' or 'file_paths'.")
        
        print(f"Initializing AdaptiveRAG with {len(texts)} text chunks...")
        
        self.classifier = QueryClassifier()
        self.strategies = {
            "Factual": FactualRetrievalStrategy(texts),
            "Analytical": AnalyticalRetrievalStrategy(texts),
            "Opinion": OpinionRetrievalStrategy(texts),
            "Contextual": ContextualRetrievalStrategy(texts)
        }
        self.llm = SimpleGeminiLLM(model="gemini-1.5-flash", temperature=0, max_tokens=4000)
        self.prompt_template = """Use the following pieces of context to answer the question at the end.        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""

    def answer(self, query: str) -> str:
        category = self.classifier.classify(query)
        strategy = self.strategies[category]
        docs = strategy.retrieve(query)
        
        # Format context and question
        context = "\n".join([doc.page_content for doc in docs])
        formatted_prompt = self.prompt_template.format(context=context, question=query)
        
        # Get response from LLM
        response = self.llm.invoke(formatted_prompt)
        return response.content


# Argument parsing functions
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run AdaptiveRAG system.")
    parser.add_argument('--texts', nargs='+', help="Input texts for retrieval")
    parser.add_argument('--files', nargs='+', help="Input files (PDF, TXT, CSV, JSON, DOCX, XLSX)")
    parser.add_argument('--directory', help="Directory containing documents to load")
    parser.add_argument('--query', help="Query to ask the system")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Determine input source
    texts = args.texts
    file_paths = None
    
    if args.directory:
        file_paths = args.directory
    elif args.files:
        file_paths = args.files
    
    # Default fallback if nothing provided
    if not texts and not file_paths:
        texts = ["The Earth is the third planet from the Sun and the only astronomical object known to harbor life."]
    
    # Initialize AdaptiveRAG
    rag_system = AdaptiveRAG(texts=texts, file_paths=file_paths)
    
    # Define test queries
    default_queries = [
        "What is the distance between the Earth and the Sun?",
        "How does the Earth's distance from the Sun affect its climate?",
        "What are the different theories about the origin of life on Earth?",
        "How does the Earth's position in the Solar System influence its habitability?"
    ]
    
    # Use provided query or default queries
    if args.query:
        queries = [args.query]
    else:
        queries = default_queries
    
    # Process queries
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        result = rag_system.answer(query)
        print(f"Answer: {result}")