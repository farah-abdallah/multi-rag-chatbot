import os
import sys
import argparse
import time
import random
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchResults
from helper_functions import encode_pdf, encode_document
import json

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path since we work with notebooks

# Load environment variables from a .env file
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)


class SimpleGeminiLLM:
    """Simple wrapper for Google Gemini API"""
    
    def __init__(self, model="gemini-1.5-flash", max_tokens=1000, temperature=0):
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


class CRAG:
    """
    A class to handle the CRAG process for document retrieval, evaluation, and knowledge refinement.
    """

    def __init__(self, file_path, model="gemini-1.5-flash", max_tokens=1000, temperature=0, lower_threshold=0.3,
                 upper_threshold=0.7):
        """
        Initializes the CRAG Retriever by encoding the document and creating the necessary models and search tools.

        Args:
            file_path (str): Path to the document file to encode (PDF, TXT, CSV, JSON, DOCX, XLSX).
            model (str): The language model to use for the CRAG process.
            max_tokens (int): Maximum tokens to use in LLM responses (default: 1000).
            temperature (float): The temperature to use for LLM responses (default: 0).
            lower_threshold (float): Lower threshold for document evaluation scores (default: 0.3).
            upper_threshold (float): Upper threshold for document evaluation scores (default: 0.7).
        """
        print("\n--- Initializing CRAG Process ---")

        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold        # Encode the document into a vector store
        self.vectorstore = encode_document(file_path)

        # Initialize Gemini language model
        self.llm = SimpleGeminiLLM(model=model, max_tokens=max_tokens, temperature=temperature)

        # Initialize search tool
        self.search = DuckDuckGoSearchResults()

    @staticmethod
    def retrieve_documents(query, faiss_index, k=3):
        docs = faiss_index.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def evaluate_documents(self, query, documents):
        return [self.retrieval_evaluator(query, doc) for doc in documents]

    def retrieval_evaluator(self, query, document):
        prompt_text = f"""On a scale from 0 to 1, how relevant is the following document to the query? 
Please respond with ONLY a number between 0 and 1, no other text.

Query: {query}
Document: {document}
Relevance score:"""
        
        result = self._call_llm_with_retry(prompt_text)
          # Extract the numeric score from the response
        try:
            score_text = result.content.strip()
            # Try to extract a number from the response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|1|0', score_text)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)  # Ensure it's between 0 and 1
            else:
                return 0.5  # Default score if no number found
        except (ValueError, AttributeError):
            return 0.5  # Default score if parsing fails
    
    def knowledge_refinement(self, document):
        prompt_text = f"""Extract the key information from the following document in bullet points:

{document}

Key points:"""
        
        result = self._call_llm_with_retry(prompt_text)
        
        # Parse the bullet points from the response
        try:
            content = result.content.strip()
            points = [line.strip() for line in content.split('\n') if line.strip()]
            return points
        except AttributeError:
            return ["Error processing document"]

    def rewrite_query(self, query):
        prompt_text = f"""Rewrite the following query to make it more suitable for a web search:

{query}

Rewritten query:"""
        
        result = self._call_llm_with_retry(prompt_text)
        
        # Extract the rewritten query from the response
        try:
            return result.content.strip()
        except AttributeError:
            return query  # Return original query if parsing fails

    @staticmethod
    def parse_search_results(results_string):
        try:
            results = json.loads(results_string)
            return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
        except json.JSONDecodeError:
            print("Error parsing search results. Returning empty list.")
            return []

    def perform_web_search(self, query):
        rewritten_query = self.rewrite_query(query)
        web_results = self.search.run(rewritten_query)
        web_knowledge = self.knowledge_refinement(web_results)
        sources = self.parse_search_results(web_results)
        return web_knowledge, sources
    
    def generate_response(self, query, knowledge, sources):
        sources_text = "\n".join([f"- {title}: {link}" if link else f"- {title}" for title, link in sources])
        
        # Determine source type for clear attribution
        source_info = ""
        if any("Retrieved document" in s[0] for s in sources) and len(sources) == 1:
            source_info = "Based on your uploaded document:"
        elif any("Retrieved document" not in s[0] for s in sources):
            source_info = "Based on web search results:"
        else:
            source_info = "Based on your uploaded document and web search results:"
        
        prompt_text = f"""Answer the following query using the provided knowledge. 
Be clear about your sources and use the exact source attribution provided.

Query: {query}

{source_info}
{knowledge}

Sources:
{sources_text}

Provide a clear, accurate answer and mention the sources appropriately. If the information comes from web search, explicitly state that. If from uploaded documents, state that clearly.

Answer:"""
        
        result = self._call_llm_with_retry(prompt_text)
        return result.content

    def run(self, query):
        print(f"\nProcessing query: {query}")

        # Retrieve and evaluate documents
        retrieved_docs = self.retrieve_documents(query, self.vectorstore)
        eval_scores = self.evaluate_documents(query, retrieved_docs)

        print(f"\nRetrieved {len(retrieved_docs)} documents")
        print(f"Evaluation scores: {eval_scores}")

        # Determine action based on evaluation scores
        max_score = max(eval_scores)
        sources = []

        if max_score > self.upper_threshold:
            print("\nAction: Correct - Using retrieved document")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = best_doc
            sources.append(("Retrieved document", ""))
        elif max_score < self.lower_threshold:
            print("\nAction: Incorrect - Performing web search")
            final_knowledge, sources = self.perform_web_search(query)
        else:
            print("\nAction: Ambiguous - Combining retrieved document and web search")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            retrieved_knowledge = self.knowledge_refinement(best_doc)
            web_knowledge, web_sources = self.perform_web_search(query)
            final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
            sources = [("Retrieved document", "")] + web_sources

        print("\nFinal knowledge:")
        print(final_knowledge)

        print("\nSources:")
        for title, link in sources:
            print(f"{title}: {link}" if link else title)

        print("\nGenerating response...")
        response = self.generate_response(query, final_knowledge, sources)
        print("\nResponse generated")
        
        # Small delay to be respectful to the API
        time.sleep(0.5)
        
        return response

    def _call_llm_with_retry(self, prompt_text, max_retries=3):
        """
        Call LLM with retry logic to handle rate limits.
        """
        for attempt in range(max_retries):
            try:
                # Add a small delay between requests to avoid rate limits
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying after {delay:.2f} seconds...")
                    time.sleep(delay)
                
                result = self.llm.invoke(prompt_text)
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                    print(f"Rate limit hit on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                else:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
        
        raise Exception("Max retries exceeded")


# Function to validate command line inputs
def validate_args(args):
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")
    if args.temperature < 0 or args.temperature > 1:
        raise ValueError("temperature must be between 0 and 1.")
    return args


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="CRAG Process for Document Retrieval and Query Answering.")
    parser.add_argument("--file_path", type=str, default="data/Understanding_Climate_Change (1).pdf",
                        help="Path to the document file to encode (supports PDF, TXT, CSV, JSON, DOCX, XLSX).")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash",
                        help="Language model to use (default: gemini-1.5-flash).")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Maximum tokens to use in LLM responses (default: 1000).")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature to use for LLM responses (default: 0).")
    parser.add_argument("--query", type=str, default="What are the main causes of climate change?",
                        help="Query to test the CRAG process.")
    parser.add_argument("--lower_threshold", type=float, default=0.3,
                        help="Lower threshold for score evaluation (default: 0.3).")
    parser.add_argument("--upper_threshold", type=float, default=0.7,
                        help="Upper threshold for score evaluation (default: 0.7).")

    return validate_args(parser.parse_args())


# Main function to handle argument parsing and call the CRAG class
def main(args):
    # Initialize the CRAG process
    crag = CRAG(
        file_path=args.file_path,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold
    )

    # Process the query
    response = crag.run(args.query)
    print(f"Query: {args.query}")
    print(f"Answer: {response}")


if __name__ == '__main__':
    main(parse_args())