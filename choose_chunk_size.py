import nest_asyncio
import random
import time
import os
import glob
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.prompts import PromptTemplate
from llama_index.core.evaluation import DatasetGenerator, FaithfulnessEvaluator, RelevancyEvaluator
import google.generativeai as genai
from llama_index.core.llms import LLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata, ChatMessage, ChatResponse
from typing import Any, Sequence, List, Union
from llama_index.core.node_parser import SentenceSplitter

# Multi-format document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, JSONLoader,
    UnstructuredWordDocumentLoader, UnstructuredExcelLoader
)

# Set up local embedding model to avoid OpenAI dependency
def setup_local_embeddings():
    """Configure local embedding model instead of OpenAI."""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Use a lightweight local embedding model
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        print("✅ Using HuggingFace local embedding model: all-MiniLM-L6-v2")
        
    except ImportError as e:
        print(f"Could not set up local embeddings: `llama-index-embeddings-huggingface` package not found, please run `pip install llama-index-embeddings-huggingface`")
        # Fallback: disable embeddings entirely for this test
        from llama_index.core.embeddings import MockEmbedding
        Settings.embed_model = MockEmbedding(embed_dim=384)
        print("Embeddings have been explicitly disabled. Using MockEmbedding.")
        print("Embeddings disabled - using basic text matching")
    except Exception as e:
        print(f"Could not set up local embeddings: {e}")
        # Fallback: disable embeddings entirely for this test
        from llama_index.core.embeddings import MockEmbedding
        Settings.embed_model = MockEmbedding(embed_dim=384)
        print("Embeddings disabled - using basic text matching")

# Apply asyncio fix for Jupyter notebooks
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Set the Google API key environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# Set up local embeddings to avoid OpenAI dependency
setup_local_embeddings()


def load_multi_format_documents(data_dir: str) -> List:
    """
    Load documents from various formats in a directory.
    Supports: PDF, TXT, CSV, JSON, DOCX, XLSX, MD, HTML
    """
    supported_extensions = ['*.pdf', '*.txt', '*.csv', '*.json', '*.docx', '*.xlsx', '*.md', '*.html']
    all_files = []
    
    # Get all supported files from directory
    for ext in supported_extensions:
        files = glob.glob(os.path.join(data_dir, ext))
        all_files.extend(files)
    
    if not all_files:
        print(f"No supported documents found in {data_dir}")
        return []
    
    all_documents = []
    
    for file_path in all_files:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            print(f"Loading {file_extension.upper()} file: {os.path.basename(file_path)}")
            
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
                print(f"Warning: Unsupported file type {file_extension}, skipping")
                continue
            
            # Load documents using LangChain loaders
            langchain_documents = loader.load()
            
            # Convert LangChain documents to LlamaIndex format
            for doc in langchain_documents:
                # Create LlamaIndex Document
                llama_doc = Document(
                    text=doc.page_content,
                    metadata={
                        **doc.metadata,
                        'source_file': os.path.basename(file_path),
                        'file_type': file_extension
                    }
                )
                all_documents.append(llama_doc)
            
            print(f"✅ Successfully loaded {len(langchain_documents)} chunks from {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
            continue
    
    print(f"📚 Total documents loaded: {len(all_documents)}")
    return all_documents


# Utility functions
def evaluate_response_time_and_accuracy(chunk_size, eval_questions, eval_documents, faithfulness_evaluator,
                                       relevancy_evaluator):
    """
    Evaluate the average response time, faithfulness, and relevancy of responses generated by Gemini for a given chunk size.

    Parameters:
    chunk_size (int): The size of data chunks being processed.
    eval_questions (list): List of evaluation questions.
    eval_documents (list): Documents used for evaluation.
    faithfulness_evaluator (FaithfulnessEvaluator): Evaluator for faithfulness.
    relevancy_evaluator (RelevancyEvaluator): Evaluator for relevancy.

    Returns:
    tuple: A tuple containing the average response time, faithfulness, and relevancy metrics.
    """
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    # Set global LLM as Gemini 1.5 Flash
    llm = GeminiLLM(model="gemini-1.5-flash")
    Settings.llm = llm
      # Create vector index with appropriate chunk overlap
    # Ensure overlap is smaller than chunk size (typically 10-20% of chunk size)
    chunk_overlap = min(20, chunk_size // 10)  # Use 10% of chunk size or 20, whichever is smaller
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_index = VectorStoreIndex.from_documents(eval_documents, transformations=[splitter])

    # Build query engine
    query_engine = vector_index.as_query_engine(similarity_top_k=5)
    num_questions = len(eval_questions)    # Iterate over each question in eval_questions to compute metrics
    for i, question in enumerate(eval_questions):
        if i > 0:  # Add small delay between requests to avoid rate limiting
            time.sleep(2)
            
        start_time = time.time()
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time

        faithfulness_result = faithfulness_evaluator.evaluate_response(response=response_vector).passing
        relevancy_result = relevancy_evaluator.evaluate_response(query=question, response=response_vector).passing

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

        print(f"  Question {i+1}/{num_questions} completed")

    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy


# Define the main class for the RAG method

class RAGEvaluator:
    def __init__(self, data_dir, num_eval_questions, chunk_sizes):
        self.data_dir = data_dir
        self.num_eval_questions = num_eval_questions
        self.chunk_sizes = chunk_sizes
        
        # Set Gemini 1.5 Flash for evaluation (initialize first)
        self.llm_gemini = GeminiLLM(model="gemini-1.5-flash")
        Settings.llm = self.llm_gemini
        
        self.documents = self.load_documents()
        self.eval_questions = self.generate_eval_questions()
        self.faithfulness_evaluator = self.create_faithfulness_evaluator()
        self.relevancy_evaluator = self.create_relevancy_evaluator()

    def load_documents(self):
        """Load documents with multi-format support"""
        print(f"🔍 Scanning directory: {self.data_dir}")
        
        try:
            # Try the new multi-format loader first
            documents = load_multi_format_documents(self.data_dir)
            if documents:
                print(f"✅ Multi-format loader successfully loaded {len(documents)} documents")
                return documents
        except Exception as e:
            print(f"⚠️ Multi-format loading failed: {e}")
        
        # Fallback to original SimpleDirectoryReader
        print("🔄 Falling back to SimpleDirectoryReader...")
        try:
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            print(f"✅ SimpleDirectoryReader loaded {len(documents)} documents")
            return documents
        except Exception as e:
            print(f"❌ Error with SimpleDirectoryReader: {e}")
            raise Exception(f"Failed to load documents from {self.data_dir}")

    def generate_eval_questions(self):
        eval_documents = self.documents[0:20]
        data_generator = DatasetGenerator.from_documents(eval_documents, llm=self.llm_gemini)
        eval_questions = data_generator.generate_questions_from_nodes()
        return random.sample(eval_questions, self.num_eval_questions)

    def create_faithfulness_evaluator(self):
        faithfulness_evaluator = FaithfulnessEvaluator(llm=self.llm_gemini)
        faithfulness_new_prompt_template = PromptTemplate("""
            Please tell if a given piece of information is directly supported by the context.
            You need to answer with either YES or NO.
            Answer YES if any part of the context explicitly supports the information, even if most of the context is unrelated. If the context does not explicitly support the information, answer NO. Some examples are provided below.
            ...
            """)
        faithfulness_evaluator.update_prompts({"your_prompt_key": faithfulness_new_prompt_template})
        return faithfulness_evaluator

    def create_relevancy_evaluator(self):
        return RelevancyEvaluator(llm=self.llm_gemini)

    def run(self):
        for chunk_size in self.chunk_sizes:
            avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(
                chunk_size,
                self.eval_questions,
                self.documents[0:20],
                self.faithfulness_evaluator,
                self.relevancy_evaluator
            )
            print(f"Chunk size {chunk_size} - Average Response time: {avg_response_time:.2f}s, "
                  f"Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
    
    def optimize_and_query(self, query: str):
        """
        Run optimization and then answer a query using the best chunk size.
        
        Args:
            query (str): The question to answer
            
        Returns:
            dict: Contains answer, optimization results, and best chunk size info
        """
        print("🔍 Running chunk size optimization...")
        
        best_chunk_size = None
        best_score = -1
        optimization_results = {}
        
        # Run optimization for each chunk size
        for chunk_size in self.chunk_sizes:
            print(f"\n📊 Testing chunk size: {chunk_size}")
            
            try:
                avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(
                    chunk_size,
                    self.eval_questions,
                    self.documents[0:20],
                    self.faithfulness_evaluator,
                    self.relevancy_evaluator
                )
                
                # Calculate composite score (higher is better)
                composite_score = (avg_faithfulness * 0.4 + avg_relevancy * 0.4 + 
                                 (1.0 / max(avg_response_time, 0.1)) * 0.2)
                
                optimization_results[chunk_size] = {
                    'response_time': avg_response_time,
                    'faithfulness': avg_faithfulness,
                    'relevancy': avg_relevancy,
                    'composite_score': composite_score
                }
                
                print(f"  ⏱️ Avg Response Time: {avg_response_time:.2f}s")
                print(f"  ✅ Avg Faithfulness: {avg_faithfulness:.2f}")
                print(f"  🎯 Avg Relevancy: {avg_relevancy:.2f}")
                print(f"  📈 Composite Score: {composite_score:.3f}")
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_chunk_size = chunk_size
                    
            except Exception as e:
                print(f"  ❌ Error testing chunk size {chunk_size}: {str(e)}")
                optimization_results[chunk_size] = {
                    'error': str(e),
                    'composite_score': 0
                }
        
        if best_chunk_size is None:
            return {
                'answer': "Error: Could not determine optimal chunk size.",
                'error': "All chunk size tests failed",
                'optimization_results': optimization_results
            }
        
        print(f"\n🎯 Best chunk size determined: {best_chunk_size} (Score: {best_score:.3f})")
        
        # Now answer the query using the best chunk size
        print(f"💬 Answering query with optimal chunk size...")
        
        try:
            # Create index with best chunk size
            chunk_overlap = min(20, best_chunk_size // 10)
            splitter = SentenceSplitter(chunk_size=best_chunk_size, chunk_overlap=chunk_overlap)
            vector_index = VectorStoreIndex.from_documents(self.documents, transformations=[splitter])
            
            # Build query engine
            query_engine = vector_index.as_query_engine(similarity_top_k=5)
            
            # Get answer
            response = query_engine.query(query)
            
            return {
                'answer': str(response),
                'best_chunk_size': best_chunk_size,
                'best_score': best_score,
                'optimization_results': optimization_results,
                'query': query
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'error': str(e),
                'best_chunk_size': best_chunk_size,
                'optimization_results': optimization_results
            }
        
    @classmethod
    def for_chatbot(cls, documents_list, chunk_sizes=None):
        """
        Create a RAGEvaluator instance specifically for chatbot use.
        
        Args:
            documents_list: List of document paths or Document objects
            chunk_sizes: List of chunk sizes to test (default: [256, 512, 1024])
            
        Returns:
            RAGEvaluator: Configured instance for chatbot use
        """
        if chunk_sizes is None:
            chunk_sizes = [256, 512, 1024]  # Reasonable defaults for chatbot
        
        # Create a temporary instance
        instance = cls.__new__(cls)
        instance.chunk_sizes = chunk_sizes
        instance.num_eval_questions = 3  # Fewer questions for faster testing
        
        # Set Gemini LLM
        instance.llm_gemini = GeminiLLM(model="gemini-1.5-flash")
        Settings.llm = instance.llm_gemini
        
        # Handle different document inputs
        if isinstance(documents_list[0], str):
            # If paths are provided, load them
            from document_augmentation import load_document_content
            from llama_index.core import Document
            
            instance.documents = []
            for doc_path in documents_list:
                try:
                    content = load_document_content(doc_path)
                    doc = Document(
                        text=content,
                        metadata={'source': doc_path, 'filename': os.path.basename(doc_path)}
                    )
                    instance.documents.append(doc)
                except Exception as e:
                    print(f"Warning: Could not load {doc_path}: {e}")
        else:
            # Assume they're already Document objects
            instance.documents = documents_list
        
        if not instance.documents:
            raise ValueError("No documents could be loaded")
        
        # Generate evaluation questions (fewer for chatbot use)
        try:
            eval_documents = instance.documents[0:min(5, len(instance.documents))]
            data_generator = DatasetGenerator.from_documents(eval_documents, llm=instance.llm_gemini)
            eval_questions = data_generator.generate_questions_from_nodes()
            instance.eval_questions = random.sample(eval_questions, min(instance.num_eval_questions, len(eval_questions)))
        except Exception as e:
            print(f"Warning: Could not generate evaluation questions: {e}")
            # Fallback to generic questions
            instance.eval_questions = [
                "What is the main topic of this document?",
                "What are the key points mentioned?",
                "Can you summarize the content?"
            ]
        
        # Create evaluators
        instance.faithfulness_evaluator = instance.create_faithfulness_evaluator()
        instance.relevancy_evaluator = instance.create_relevancy_evaluator()
        
        return instance
        


# Argument Parsing

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='RAG Method Evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory of the documents')
    parser.add_argument('--num_eval_questions', type=int, default=5, help='Number of evaluation questions (reduced for rate limits)')
    parser.add_argument('--chunk_sizes', nargs='+', type=int, default=[128, 256], help='List of chunk sizes')
    return parser.parse_args()


# Gemini LLM wrapper for LlamaIndex
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"
    client: Any = None
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None, **kwargs):
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        client = genai.GenerativeModel(model)
        super().__init__(model=model, client=client, **kwargs)
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=32768,
            num_output=8192,
            model_name=self.model,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        def _make_request():
            response = self.client.generate_content(prompt)
            return CompletionResponse(text=response.text)
        
        return make_request_with_retry(_make_request)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        response = self.complete(prompt, **kwargs)
        yield response
    
    @llm_completion_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Convert chat messages to a single prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        response = self.client.generate_content(prompt)
        return ChatResponse(message=ChatMessage(role="assistant", content=response.text))
    
    @llm_completion_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        response = self.chat(messages, **kwargs)
        yield response
    
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # For simplicity, use sync version
        return self.complete(prompt, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs: Any):
        response = await self.acomplete(prompt, **kwargs)
        yield response
    
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # For simplicity, use sync version
        return self.chat(messages, **kwargs)
    
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        response = await self.achat(messages, **kwargs)
        yield response


def make_request_with_retry(func, max_retries=3, base_delay=1):
    """Make API request with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                if attempt < max_retries - 1:
                    # Extract retry delay from error message if available
                    retry_delay = 25  # Default from error message
                    if "retry_delay" in str(e):
                        try:
                            import re
                            delay_match = re.search(r'seconds: (\d+)', str(e))
                            if delay_match:
                                retry_delay = int(delay_match.group(1))
                        except:
                            pass
                    
                    wait_time = max(retry_delay, base_delay * (2 ** attempt)) + random.uniform(0, 2)
                    print(f"Rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts. Please wait and try again later.")
                    raise e
            else:                raise e
    return None


if __name__ == "__main__":
    args = parse_args()
    evaluator = RAGEvaluator(data_dir=args.data_dir, num_eval_questions=args.num_eval_questions,
                             chunk_sizes=args.chunk_sizes)
    evaluator.run()