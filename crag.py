import os
import sys
import argparse
import time
import random
import io
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchResults
from helper_functions import encode_pdf, encode_document
import json
import requests
from urllib.parse import quote
import streamlit as st

# Import API key manager for rotation
try:
    from api_key_manager import get_api_manager
    API_MANAGER_AVAILABLE = True
    print("🔑 CRAG API key rotation manager loaded")
except ImportError:
    API_MANAGER_AVAILABLE = False
    print("⚠️ CRAG API key manager not found - using single key mode")

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path since we work with notebooks

# Load environment variables from a .env file
load_dotenv()

# Set up API key management
if API_MANAGER_AVAILABLE:
    try:
        api_manager = get_api_manager()
        print(f"🎯 CRAG API Manager Status: {api_manager.get_status()}")
    except Exception as e:
        print(f"⚠️ CRAG API Manager initialization failed: {e}")
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            print("🔑 CRAG using fallback single API key")
else:
    # Fallback to single key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        print("🔑 CRAG using single API key mode")


class SimpleGeminiLLM:
    """Enhanced wrapper for Google Gemini API with Key Rotation"""
    
    def __init__(self, model="gemini-1.5-flash", max_tokens=1000, temperature=0):
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
          # Set up API manager if available
        if API_MANAGER_AVAILABLE:
            try:
                self.api_manager = get_api_manager()
                self.use_rotation = True
            except:
                self.api_manager = None
                self.use_rotation = False
        else:
            self.api_manager = None
            self.use_rotation = False
        
        self.model = genai.GenerativeModel(model)
    
    def invoke(self, prompt_text, max_retries=3):
        """Invoke the model with automatic key rotation on quota errors"""
        for attempt in range(max_retries):
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
                error_msg = str(e)
                
                # Check if it's a quota error and we have API rotation
                if self.use_rotation and self.api_manager and self.api_manager.handle_quota_error(error_msg):
                    # Key switched, reinitialize model with new key
                    self.model = genai.GenerativeModel(self.model_name)
                    print(f"🔄 CRAG retrying with new API key... (attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    # Non-quota error or no rotation available
                    if attempt < max_retries - 1:
                        print(f"❌ CRAG API Error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                        continue
                    else:
                        raise Exception(f"Gemini API error: {error_msg}")
        
        # All retries exhausted
        raise Exception(f"CRAG: All API attempts exhausted after {max_retries} attempts")
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
        self.upper_threshold = upper_threshold
        
        # Load configuration from environment variables
        self.web_search_enabled = os.getenv('CRAG_WEB_SEARCH', 'true').lower() == 'true'
        self.fallback_mode = os.getenv('CRAG_FALLBACK_MODE', 'true').lower() == 'true'
        
        print(f"Web search enabled: {self.web_search_enabled}")
        print(f"Fallback mode enabled: {self.fallback_mode}")
        
        # Encode the document into a vector store
        self.vectorstore = encode_document(file_path)

        # Initialize Gemini language model
        self.llm = SimpleGeminiLLM(model=model, max_tokens=max_tokens, temperature=temperature)

        # Initialize search tool only if web search is enabled
        if self.web_search_enabled:
            self.search = DuckDuckGoSearchResults()
        else:
            self.search = None
            print("Web search disabled via configuration")

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
        """Improved web search with proper error handling and fallback"""
        # Check if web search is disabled
        if not self.web_search_enabled:
            print("Web search disabled via configuration")
            return ["Web search disabled"], [("Configuration", "")]
        
        try:
            # Try improved web search first
            web_results = self.safe_web_search(query)
            
            if web_results:
                # Process successful results
                web_knowledge = []
                sources = []
                
                for result in web_results:
                    if isinstance(result, dict):
                        title = result.get('title', 'Untitled')
                        content = result.get('snippet', result.get('content', ''))
                        link = result.get('link', result.get('url', ''))
                        
                        if content:
                            web_knowledge.append(content)
                            sources.append((title, link))
                
                if web_knowledge:
                    return web_knowledge, sources
            
            # Fallback to original method if improved search fails
            if self.search:  # Only if search tool is available
                print("Falling back to original web search...")
                rewritten_query = self.rewrite_query(query)
                web_results = self.search.run(rewritten_query)
                web_knowledge = self.knowledge_refinement(web_results)
                sources = self.parse_search_results(web_results)
                return web_knowledge, sources
            else:
                print("No search tool available")
                return ["Web search unavailable"], [("No Search Tool", "")]
            
        except Exception as e:
            print(f"Web search failed: {str(e)}")
            # Return empty results instead of crashing
            return ["Web search unavailable at this time."], [("Web Search Error", "")]
    
    def safe_web_search(self, query, max_results=3):
        """Improved web search with proper error handling"""
        try:
            # Clean and encode the query
            clean_query = query.strip().replace('\n', ' ')[:200]  # Limit length
            encoded_query = quote(clean_query)
            
            # Use proper headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Referer': 'https://duckduckgo.com/',
            }
            
            # Try DuckDuckGo with improved parameters
            url = "https://html.duckduckgo.com/html"
            params = {
                'q': clean_query,
                'b': '',
                'kl': 'us-en',  # Change from 'wt-wt' to avoid issues
                'df': 'y'
            }
            
            print(f"Searching for: {clean_query}")
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200 and response.content:
                # Parse results from HTML response
                results = self._parse_duckduckgo_html(response.text)
                if results:
                    print(f"Found {len(results)} web results")
                    return results[:max_results]
                else:
                    print("No results found in response")
                    return []
            else:
                print(f"Web search failed: Status {response.status_code}")
                return []
                
        except requests.RequestException as e:
            print(f"Network error during web search: {e}")
            return []
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    def _parse_duckduckgo_html(self, html_content):
        """Parse DuckDuckGo HTML results"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            results = []
            # Look for result containers
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:5]:  # Limit to first 5 results
                try:
                    # Extract title
                    title_link = div.find('a', class_='result__a')
                    title = title_link.get_text(strip=True) if title_link else "Untitled"
                    link = title_link.get('href', '') if title_link else ''
                    
                    # Extract snippet
                    snippet_elem = div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if title and snippet:
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'link': link
                        })
                except Exception as e:
                    print(f"Error parsing individual result: {e}")
                    continue
            
            return results
            
        except ImportError:
            print("BeautifulSoup not available, falling back to simple parsing")
            # Simple fallback parsing without BeautifulSoup
            return self._simple_parse_html(html_content)
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return []
    
    def _simple_parse_html(self, html_content):
        """Simple HTML parsing without BeautifulSoup"""
        try:
            import re
            
            # Simple regex to extract basic information
            title_pattern = r'<a[^>]*class="result__a"[^>]*>([^<]+)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'
            
            titles = re.findall(title_pattern, html_content)
            snippets = re.findall(snippet_pattern, html_content)
            
            results = []
            for i, (title, snippet) in enumerate(zip(titles[:3], snippets[:3])):
                results.append({
                    'title': title.strip(),
                    'snippet': snippet.strip(),
                    'link': ''
                })
            
            return results
            
        except Exception as e:
            print(f"Simple parsing failed: {e}")
            return []
    
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
        """CRAG with graceful web search fallback"""
        print(f"\nProcessing query: {query}")

        try:
            # Step 1: Retrieve and evaluate documents
            retrieved_docs = self.retrieve_documents(query, self.vectorstore)
            eval_scores = self.evaluate_documents(query, retrieved_docs)

            print(f"\nRetrieved {len(retrieved_docs)} documents")
            print(f"Evaluation scores: {eval_scores}")

            # Determine action based on evaluation scores
            max_score = max(eval_scores) if eval_scores else 0.0
            sources = []

            if max_score > self.upper_threshold:
                print("\nAction: Correct - Using retrieved document")
                best_doc = retrieved_docs[eval_scores.index(max_score)]
                final_knowledge = best_doc
                sources.append(("Retrieved document", ""))
                
            elif max_score < self.lower_threshold:
                print("\nAction: Incorrect - Performing web search")
                try:
                    final_knowledge, sources = self.perform_web_search(query)
                    
                    # Check if web search actually returned useful results
                    if not final_knowledge or final_knowledge == ["Web search unavailable at this time."]:
                        print("\nWeb search failed, falling back to best local document")
                        if retrieved_docs:
                            best_doc = retrieved_docs[0]  # Use first document as fallback
                            final_knowledge = best_doc
                            sources = [("Retrieved document (fallback)", "")]
                            try:
                                st.warning("🌐 Web search failed, using local documents instead")
                            except:
                                print("Warning: Web search failed, using local documents instead")
                        else:
                            final_knowledge = "No relevant information found in documents or web search."
                            sources = [("No sources available", "")]
                            
                except Exception as web_error:
                    print(f"\nWeb search error: {str(web_error)}")
                    # Fallback to local documents
                    if retrieved_docs:
                        best_doc = retrieved_docs[0]
                        final_knowledge = best_doc
                        sources = [("Retrieved document (web search failed)", "")]
                        try:
                            st.warning(f"🌐 Web search failed: {str(web_error)[:50]}... Using local documents.")
                        except:
                            print(f"Warning: Web search failed: {str(web_error)[:50]}... Using local documents.")
                    else:
                        final_knowledge = f"Web search failed and no local documents available. Error: {str(web_error)}"
                        sources = [("Error", "")]
                        
            else:
                print("\nAction: Ambiguous - Combining retrieved document and web search")
                best_doc = retrieved_docs[eval_scores.index(max_score)]
                retrieved_knowledge = self.knowledge_refinement(best_doc)
                
                try:
                    web_knowledge, web_sources = self.perform_web_search(query)
                    
                    # Check if web search was successful
                    if web_knowledge and web_knowledge != ["Web search unavailable at this time."]:
                        final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
                        sources = [("Retrieved document", "")] + web_sources
                    else:
                        print("\nWeb search failed in ambiguous case, using only retrieved document")
                        final_knowledge = "\n".join(retrieved_knowledge)
                        sources = [("Retrieved document", "")]
                        try:
                            st.info("ℹ️ Web search unavailable, using only local documents")
                        except:
                            print("Info: Web search unavailable, using only local documents")
                            
                except Exception as web_error:
                    print(f"\nWeb search error in ambiguous case: {str(web_error)}")
                    final_knowledge = "\n".join(retrieved_knowledge)
                    sources = [("Retrieved document", "")]
                    try:
                        st.info(f"ℹ️ Web search failed: {str(web_error)[:50]}... Using only local documents.")
                    except:
                        print(f"Info: Web search failed: {str(web_error)[:50]}... Using only local documents.")

            print("\nFinal knowledge:")
            print(final_knowledge[:500] + "..." if len(str(final_knowledge)) > 500 else str(final_knowledge))

            print("\nSources:")
            for title, link in sources:
                print(f"{title}: {link}" if link else title)

            print("\nGenerating response...")
            try:
                response = self.generate_response(query, final_knowledge, sources)
                print("\nResponse generated successfully")
            except Exception as response_error:
                print(f"\nError generating response: {str(response_error)}")
                response = f"Error generating response: {str(response_error)}"
            
            # Small delay to be respectful to the API
            time.sleep(0.5)
            
            return response
            
        except Exception as e:
            print(f"\nCRAG process failed: {str(e)}")
            # Final fallback
            try:
                if hasattr(self, 'vectorstore'):
                    docs = self.vectorstore.similarity_search(query, k=1)
                    if docs:
                        return f"CRAG failed, but found this relevant information: {docs[0].page_content[:500]}..."
                return f"CRAG process failed: {str(e)}"
            except:
                return f"CRAG process failed: {str(e)}"

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