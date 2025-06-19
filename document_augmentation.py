import sys
import os
import re
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from enum import Enum
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import Any, Dict, List, Tuple
import argparse
import glob

# Multi-format document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, JSONLoader,
    UnstructuredWordDocumentLoader, UnstructuredExcelLoader
)

from dotenv import load_dotenv

load_dotenv()

# Configure Google Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path

from helper_functions import *


class QuestionGeneration(Enum):
    """
    Enum class to specify the level of question generation for document processing.
    """
    DOCUMENT_LEVEL = 1
    FRAGMENT_LEVEL = 2


DOCUMENT_MAX_TOKENS = 4000
DOCUMENT_OVERLAP_TOKENS = 100
FRAGMENT_MAX_TOKENS = 128
FRAGMENT_OVERLAP_TOKENS = 16
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL
QUESTIONS_PER_DOCUMENT = 40


# Configure Gemini API
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


class SentenceTransformerEmbeddings:
    """
    Custom embeddings wrapper using sentence-transformers directly
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text])
        return embedding[0].tolist()
    
    def __call__(self, query: str) -> List[float]:
        return self.embed_query(query)


def clean_and_filter_questions(questions: List[str]) -> List[str]:
    cleaned_questions = []
    for question in questions:
        cleaned_question = re.sub(r'^\d+\.\s*', '', question.strip())
        if cleaned_question.endswith('?'):
            cleaned_questions.append(cleaned_question)
    return cleaned_questions


def generate_questions(text: str) -> List[str]:
    """Default question generation for backward compatibility"""
    prompt_text = f"""Based on the following text, generate {QUESTIONS_PER_DOCUMENT} diverse questions that could be answered using this content.

Text content:
{text[:2000]}...

Generate {QUESTIONS_PER_DOCUMENT} questions, one per line, numbered 1-{QUESTIONS_PER_DOCUMENT}:
"""
    
    try:
        # Use direct Gemini SDK
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_text)
        return parse_questions_from_response(response.text, QUESTIONS_PER_DOCUMENT)
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


def generate_answer(content: str, question: str) -> str:
    prompt_text = f"""Based on the following context, provide a clear and accurate answer to the question.

Context:
{content}

Question: {question}

Answer:"""
    
    try:
        # Use direct Gemini SDK
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error generating answer: {e}"


def split_document(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    tokens = re.findall(r'\b\w+\b', document)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(chunk_tokens)
        if i + chunk_size >= len(tokens):
            break
    return [" ".join(chunk) for chunk in chunks]


def print_document(comment: str, document: Any) -> None:
    print(f'{comment} (type: {document.metadata["type"]}, index: {document.metadata["index"]}): {document.page_content}')


class DocumentProcessor:
    def __init__(self, content: str, embedding_model: SentenceTransformerEmbeddings, file_path: str = None):
        self.content = content
        self.embedding_model = embedding_model
        self.file_path = file_path

    def run(self):
        text_documents = split_document(self.content, DOCUMENT_MAX_TOKENS, DOCUMENT_OVERLAP_TOKENS)
        print(f'Text content split into: {len(text_documents)} documents')

        documents = []
        counter = 0
        for i, text_document in enumerate(text_documents):
            text_fragments = split_document(text_document, FRAGMENT_MAX_TOKENS, FRAGMENT_OVERLAP_TOKENS)
            print(f'Text document {i} - split into: {len(text_fragments)} fragments')

            for j, text_fragment in enumerate(text_fragments):
                documents.append(Document(
                    page_content=text_fragment,
                    metadata={"type": "ORIGINAL", "index": counter, "text": text_document}
                ))
                counter += 1

                if QUESTION_GENERATION == QuestionGeneration.FRAGMENT_LEVEL:
                    if self.file_path:
                        questions = detect_file_format_and_generate_questions(text_fragment, self.file_path, QUESTIONS_PER_DOCUMENT)
                    else:
                        questions = generate_questions(text_fragment)
                    documents.extend([
                        Document(page_content=question,
                                 metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                        for idx, question in enumerate(questions)
                    ])
                    counter += len(questions)
                    print(f'Text document {i} Text fragment {j} - generated: {len(questions)} questions')

            if QUESTION_GENERATION == QuestionGeneration.DOCUMENT_LEVEL:
                if self.file_path:
                    questions = detect_file_format_and_generate_questions(text_document, self.file_path, QUESTIONS_PER_DOCUMENT)
                else:
                    questions = generate_questions(text_document)
                documents.extend([
                    Document(page_content=question,
                             metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                    for idx, question in enumerate(questions)
                ])
                counter += len(questions)
                print(f'Text document {i} - generated: {len(questions)} questions ({os.path.splitext(self.file_path or "unknown")[1].upper()} format)')

        for document in documents:
            print_document("Dataset", document)

        print(f'Creating store, calculating embeddings for {len(documents)} FAISS documents')
        vectorstore = FAISS.from_documents(documents, self.embedding_model)

        print("Creating retriever returning the most relevant FAISS document")
        return vectorstore.as_retriever(search_kwargs={"k": 1})


# Replace the parse_args function:

def parse_args():
    parser = argparse.ArgumentParser(description="Document Augmentation with RAG - Multi-Format Support")
    parser.add_argument('--path', type=str, default='./data/Understanding_Climate_Change (1).pdf',
                       help='Path to the document file (supports PDF, TXT, CSV, JSON, DOCX, XLSX)')
    parser.add_argument('--query', type=str, required=False,
                       help='Query to ask about the document')
    parser.add_argument('--num_questions', type=int, default=5,
                       help='Number of synthetic questions to generate')
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Size of text chunks')
    parser.add_argument('--chunk_overlap', type=int, default=200,
                       help='Overlap between text chunks')
    return parser.parse_args()

def load_document_content(file_path: str) -> str:
    """
    Load content from various document formats and return as a single string.
    Supports: PDF, TXT, CSV, JSON, DOCX, XLSX
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        print(f"📄 Loading {file_extension.upper()} file: {os.path.basename(file_path)}")
        
        # Choose appropriate loader based on file type
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.txt', '.md']:
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
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load documents
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No content could be extracted from {file_path}")
        
        # Combine all document content into a single string
        content_parts = []
        for doc in documents:
            if doc.page_content.strip():  # Only add non-empty content
                content_parts.append(doc.page_content.strip())
        
        combined_content = "\n\n".join(content_parts)
        
        print(f"✅ Successfully loaded {len(documents)} pages/chunks from {os.path.basename(file_path)}")
        print(f"📊 Total content length: {len(combined_content):,} characters")
        
        return combined_content
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {str(e)}")
        
        # Fallback to the original PDF-only method if it's a PDF
        if file_path.lower().endswith('.pdf'):
            try:
                print("🔄 Falling back to original PDF loader...")
                from helper_functions import read_pdf_to_string
                return read_pdf_to_string(file_path)
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
        
        raise Exception(f"Could not load document: {e}")


def detect_file_format_and_generate_questions(content: str, file_path: str, num_questions: int) -> List[str]:
    """
    Generate format-specific questions based on the document type.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Format-specific question generation
    if file_extension == '.csv':
        return generate_data_questions(content, num_questions)
    elif file_extension == '.json':
        return generate_structural_questions(content, num_questions)
    elif file_extension in ['.pdf', '.docx']:
        return generate_conceptual_questions(content, num_questions)
    elif file_extension in ['.txt', '.md']:
        return generate_explanatory_questions(content, num_questions)
    else:
        # Default to general question generation
        return generate_questions(content)


def generate_data_questions(content: str, num_questions: int) -> List[str]:
    """Generate data-centric questions for CSV and structured data"""
    prompt_text = f"""Based on the following CSV/structured data, generate {num_questions} data-focused questions that can be answered using this data.

Focus on:
- Specific values and comparisons
- Trends and patterns
- Categorical breakdowns
- Statistical information

Data content:
{content[:2000]}...

Generate {num_questions} data-focused questions, one per line, numbered 1-{num_questions}:
"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_text)
        return parse_questions_from_response(response.text, num_questions)
    except Exception as e:
        print(f"Error generating data questions: {e}")
        return []


def generate_structural_questions(content: str, num_questions: int) -> List[str]:
    """Generate structure/configuration questions for JSON and structured documents"""
    prompt_text = f"""Based on the following JSON/structured content, generate {num_questions} questions about structure, configuration, and relationships.

Focus on:
- What are the key components?
- How are things configured?
- What are the relationships between elements?
- What are the available options/settings?

Content:
{content[:2000]}...

Generate {num_questions} structural questions, one per line, numbered 1-{num_questions}:
"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_text)
        return parse_questions_from_response(response.text, num_questions)
    except Exception as e:
        print(f"Error generating structural questions: {e}")
        return []


def generate_conceptual_questions(content: str, num_questions: int) -> List[str]:
    """Generate conceptual questions for PDF and complex documents"""
    prompt_text = f"""Based on the following content, generate {num_questions} conceptual questions covering different levels of understanding.

Include:
- Basic definitions (What is...?)
- Causal relationships (How does...? Why does...?)
- Applications (How can...? When should...?)
- Analysis (What are the implications...?)

Content:
{content[:2000]}...

Generate {num_questions} conceptual questions, one per line, numbered 1-{num_questions}:
"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_text)
        return parse_questions_from_response(response.text, num_questions)
    except Exception as e:
        print(f"Error generating conceptual questions: {e}")
        return []


def generate_explanatory_questions(content: str, num_questions: int) -> List[str]:
    """Generate explanatory questions for text documents"""
    prompt_text = f"""Based on the following text content, generate {num_questions} explanatory questions that help understand the content.

Focus on:
- Key concepts and definitions
- Processes and procedures
- Benefits and advantages
- Examples and applications

Content:
{content[:2000]}...

Generate {num_questions} explanatory questions, one per line, numbered 1-{num_questions}:
"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_text)
        return parse_questions_from_response(response.text, num_questions)
    except Exception as e:
        print(f"Error generating explanatory questions: {e}")
        return []


def parse_questions_from_response(response_text: str, num_questions: int) -> List[str]:
    """Parse questions from LLM response text"""
    questions = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for numbered questions (1., 2., etc. or 1), 2), etc.)
        if any(line.startswith(f"{i}.") or line.startswith(f"{i}") for i in range(1, num_questions + 1)):
            # Remove the number and clean up
            question = line.split('.', 1)[-1].split(')', 1)[-1].strip()
            if question and len(question) > 10 and question.endswith('?'):
                questions.append(question)
    
    return list(set(questions[:num_questions]))


if __name__ == "__main__":
    args = parse_args()

    # Load document using multi-format support
    print(f"🚀 Starting Document Augmentation with multi-format support")
    print(f"📁 Processing file: {args.path}")
    
    # Detect file type
    file_extension = os.path.splitext(args.path)[1].lower()
    supported_formats = ['.pdf', '.txt', '.csv', '.json', '.docx', '.xlsx', '.md']
    
    if file_extension not in supported_formats:
        print(f"❌ Unsupported file format: {file_extension}")
        print(f"✅ Supported formats: {', '.join(supported_formats)}")
        sys.exit(1)
    
    # Load document content using multi-format loader
    content = load_document_content(args.path)
    
    # Instantiate SentenceTransformer Embeddings class that will be used by FAISS
    print("🔧 Initializing embeddings model...")
    embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Process documents and create retriever with file path for format-specific processing
    print("🔄 Processing document with format-specific augmentation...")
    processor = DocumentProcessor(content, embedding_model, args.path)
    document_query_retriever = processor.run()

    # Example usage of the retriever
    print("\n" + "="*60)
    print("🎯 TESTING DOCUMENT AUGMENTATION")
    print("="*60)
    
    query = "What is climate change?"
    retrieved_docs = document_query_retriever.get_relevant_documents(query)
    print(f"\n🔍 Query: {query}")
    print(f"📖 Retrieved document: {retrieved_docs[0].page_content}")

    # Further query example
    query = "How do freshwater ecosystems change due to alterations in climatic factors?"
    retrieved_documents = document_query_retriever.get_relevant_documents(query)
    for doc in retrieved_documents:
        print_document("Relevant fragment retrieved", doc)

    context = doc.metadata['text']
    answer = generate_answer(context, query)
    print(f'\n🎯 Answer:\n{"-"*40}\n{answer}\n{"-"*40}')
    
    print(f"\n✅ Document augmentation complete for {file_extension.upper()} format!")