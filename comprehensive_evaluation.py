"""
Comprehensive RAG Technique Evaluation Framework
Evaluates all RAG techniques across multiple documents and metrics
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Import RAG systems
from adaptive_rag import AdaptiveRAG
from crag import CRAG
from document_augmentation import DocumentProcessor, SentenceTransformerEmbeddings
from explainable_retrieval import ExplainableRAGMethod
from helper_functions import encode_document

# Import evaluation metrics
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class RAGEvaluator:
    def __init__(self, test_documents_dir: str = "data/"):
        self.test_documents_dir = test_documents_dir
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.results = {}
        
    def load_test_documents(self) -> List[str]:
        """Load all test documents from the directory"""
        supported_extensions = ['.pdf', '.txt', '.csv', '.json', '.docx', '.xlsx']
        documents = []
        
        for ext in supported_extensions:
            documents.extend(Path(self.test_documents_dir).glob(f'*{ext}'))
        
        return [str(doc) for doc in documents]
    
    def load_document_content(self, file_path: str) -> str:
        """Load document content based on file type"""
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.to_string()
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            elif file_path.endswith('.pdf'):
                # Use existing PDF processing
                from helper_functions import encode_pdf
                vectorstore = encode_pdf(file_path)
                docs = vectorstore.similarity_search("", k=10)  # Get all chunks
                return "\n".join([doc.page_content for doc in docs])
            else:
                return "Unsupported file type"
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return "Error loading document"
    
    def generate_test_queries(self, document_content: str, num_queries: int = 8) -> List[str]:
        """Generate diverse test queries for a document"""
        content_preview = document_content[:3000]  # Use first 3000 chars
        
        prompt = f"""Based on the following document content, generate {num_queries} diverse questions that test different aspects of information retrieval. Create questions that cover:

1. Factual questions (What, Who, When, Where)
2. Analytical questions (How, Why, Compare)
3. Conceptual questions (Explain, Define)
4. Application questions (How can, What if)
5. Numerical/data questions (if applicable)
6. Process/procedure questions (if applicable)

Document content:
{content_preview}

Generate exactly {num_queries} diverse questions, one per line:"""
        
        try:
            response = self.model.generate_content(prompt)
            questions = [q.strip() for q in response.text.split('\n') if q.strip() and '?' in q]
            
            # Ensure we have the right number of questions
            if len(questions) < num_queries:
                # Add generic questions if needed
                generic_questions = [
                    "What is the main topic of this document?",
                    "What are the key points mentioned?",
                    "How does this information relate to the broader context?",
                    "What are the implications of the information presented?",
                    "Can you summarize the main findings?",
                    "What are the most important details?",
                    "How is this information structured?",
                    "What conclusions can be drawn?"
                ]
                questions.extend(generic_questions[:num_queries - len(questions)])
            
            return questions[:num_queries]
            
        except Exception as e:
            print(f"Error generating queries: {e}")
            return [
                "What is the main topic of this document?",
                "What are the key points mentioned?",
                "How does this information relate to the broader context?",
                "What are the implications of the information presented?",
                "Can you summarize the main findings?"
            ][:num_queries]
    
    def evaluate_with_llm(self, metric_name: str, query: str, answer: str, context: str = "") -> float:
        """Generic LLM-based evaluation"""
        
        if metric_name == "faithfulness":
            prompt = f"""Evaluate how faithful this answer is to the provided context. Rate from 0 to 1.
            
Context: {context[:1500]}
Query: {query}
Answer: {answer}

Does the answer accurately reflect the information in the context without adding false information? 
Respond with only a number between 0 and 1:"""

        elif metric_name == "relevance":
            prompt = f"""Evaluate how relevant this answer is to the query. Rate from 0 to 1.

Query: {query}
Answer: {answer}

How well does the answer address the specific question asked? 
Respond with only a number between 0 and 1:"""

        elif metric_name == "completeness":
            prompt = f"""Evaluate how complete this answer is given the query and available context. Rate from 0 to 1.

Context: {context[:1500]}
Query: {query}
Answer: {answer}

Does the answer provide comprehensive coverage of the question? 
Respond with only a number between 0 and 1:"""

        elif metric_name == "clarity":
            prompt = f"""Evaluate how clear and well-structured this answer is. Rate from 0 to 1.

Answer: {answer}

Is the answer clear, well-organized, and easy to understand? 
Respond with only a number between 0 and 1:"""
        
        try:
            response = self.model.generate_content(prompt)
            score_text = response.text.strip()
            
            # Extract numeric score
            import re
            numbers = re.findall(r'0\.\d+|1\.0|1|0', score_text)
            if numbers:
                score = float(numbers[0])
                return max(0, min(1, score))
            else:
                return 0.5
                
        except Exception as e:
            print(f"Error evaluating {metric_name}: {e}")
            return 0.5
    
    def load_rag_system(self, technique: str, document_path: str):
        """Load a specific RAG system"""
        try:
            print(f"   Loading {technique}...")
            
            if technique == "Adaptive RAG":
                return AdaptiveRAG(file_paths=[document_path])
                
            elif technique == "CRAG":
                return CRAG(document_path)
                
            elif technique == "Document Augmentation":
                content = self.load_document_content(document_path)
                embedding_model = SentenceTransformerEmbeddings()
                processor = DocumentProcessor(content, embedding_model, document_path)
                return processor.run()
                
            elif technique == "Explainable Retrieval":
                texts = [self.load_document_content(document_path)]
                return ExplainableRAGMethod(texts)
                
            elif technique == "Basic RAG":
                return encode_document(document_path)
                
        except Exception as e:
            print(f"   ❌ Error loading {technique}: {e}")
            return None
    
    def get_rag_response(self, technique: str, rag_system, query: str) -> Tuple[str, str, float]:
        """Get response from RAG system with timing and context extraction"""
        start_time = time.time()
        context = ""
        
        try:
            if technique == "Adaptive RAG":
                response = rag_system.answer(query)
                # Extract context from adaptive RAG
                try:
                    # Get context from the factual strategy as a default
                    docs = rag_system.strategies["Factual"].db.similarity_search(query, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                except:
                    context = "Context extraction failed"
                    
            elif technique == "CRAG":
                response = rag_system.run(query)
                # Extract context from CRAG
                try:
                    docs = rag_system.vectorstore.similarity_search(query, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                except:
                    context = "Context extraction failed"
                    
            elif technique == "Document Augmentation":
                # Document Augmentation uses different interface
                try:
                    docs = rag_system.get_relevant_documents(query)
                    if docs:
                        context = docs[0].page_content if hasattr(docs[0], 'page_content') else str(docs[0])
                        # Generate response using the context
                        response_prompt = f"Based on this context, answer the question: {query}\n\nContext: {context}"
                        response_result = self.model.generate_content(response_prompt)
                        response = response_result.text
                    else:
                        response = "No relevant documents found."
                        context = "No context available"
                except Exception as e:
                    response = f"Document Augmentation error: {e}"
                    context = "Error retrieving context"
                    
            elif technique == "Explainable Retrieval":
                response = rag_system.answer(query)
                # Get context from explainable retrieval
                try:
                    detailed_results = rag_system.run(query)
                    if detailed_results and isinstance(detailed_results, list):
                        context = "\n".join([result.get('content', '') for result in detailed_results])
                    else:
                        context = "Context extraction failed"
                except:
                    context = "Context extraction failed"
                    
            elif technique == "Basic RAG":
                docs = rag_system.similarity_search(query, k=3)
                if docs:
                    context = "\n".join([doc.page_content for doc in docs])
                    # Generate response using basic method
                    response_prompt = f"Based on these documents, answer the question: {query}\n\nDocuments: {context[:1000]}"
                    response_result = self.model.generate_content(response_prompt)
                    response = response_result.text
                else:
                    response = "No relevant documents found."
                    context = "No context available"
            
            response_time = time.time() - start_time
            return response, context, response_time
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Error with {technique}: {str(e)}"
            print(f"   ⚠️ {error_msg}")
            return error_msg, "", response_time
    
    def evaluate_single_document(self, document_path: str, num_queries: int = 6) -> Dict:
        """Evaluate all RAG techniques on a single document"""
        doc_name = os.path.basename(document_path)
        print(f"\n{'='*60}")
        print(f"📄 Evaluating document: {doc_name}")
        print('='*60)
        
        # Load document content for query generation
        try:
            content = self.load_document_content(document_path)
            if not content or content == "Error loading document":
                print(f"❌ Could not load document content for {doc_name}")
                return {}
        except Exception as e:
            print(f"❌ Error loading document: {e}")
            return {}
        
        # Generate test queries
        print("🔄 Generating test queries...")
        queries = self.generate_test_queries(content, num_queries)
        print(f"✅ Generated {len(queries)} test queries:")
        for i, q in enumerate(queries, 1):
            print(f"   {i}. {q}")
        
        techniques = ["Adaptive RAG", "CRAG", "Document Augmentation", "Explainable Retrieval", "Basic RAG"]
        results = {}
        
        for technique in techniques:
            print(f"\n🔧 Testing {technique}...")
            
            # Load RAG system
            rag_system = self.load_rag_system(technique, document_path)
            if rag_system is None:
                print(f"❌ Failed to load {technique}")
                continue
            
            technique_results = {
                'responses': [],
                'contexts': [],
                'response_times': [],
                'faithfulness_scores': [],
                'relevance_scores': [],
                'completeness_scores': [],
                'clarity_scores': [],
                'queries': queries.copy()
            }
            
            for i, query in enumerate(queries, 1):
                print(f"   Query {i}/{len(queries)}: {query[:60]}...")
                
                # Get response
                response, context, response_time = self.get_rag_response(technique, rag_system, query)
                
                # Skip evaluation if there was an error
                if response.startswith("Error"):
                    print(f"      ❌ Skipping evaluation due to error")
                    continue
                
                # Evaluate metrics with small delays
                print(f"      📊 Evaluating metrics...")
                faithfulness = self.evaluate_with_llm("faithfulness", query, response, context)
                time.sleep(0.5)  # Rate limiting
                
                relevance = self.evaluate_with_llm("relevance", query, response)
                time.sleep(0.5)
                
                completeness = self.evaluate_with_llm("completeness", query, response, context)
                time.sleep(0.5)
                
                clarity = self.evaluate_with_llm("clarity", query, response)
                time.sleep(0.5)
                
                # Store results
                technique_results['responses'].append(response)
                technique_results['contexts'].append(context)
                technique_results['response_times'].append(response_time)
                technique_results['faithfulness_scores'].append(faithfulness)
                technique_results['relevance_scores'].append(relevance)
                technique_results['completeness_scores'].append(completeness)
                technique_results['clarity_scores'].append(clarity)
                
                print(f"      ⚡ Time: {response_time:.2f}s | 🎯 Relevance: {relevance:.2f} | ✅ Faithfulness: {faithfulness:.2f}")
            
            # Calculate aggregate metrics
            if technique_results['faithfulness_scores']:  # Only if we have scores
                technique_results['avg_faithfulness'] = np.mean(technique_results['faithfulness_scores'])
                technique_results['avg_relevance'] = np.mean(technique_results['relevance_scores'])
                technique_results['avg_completeness'] = np.mean(technique_results['completeness_scores'])
                technique_results['avg_clarity'] = np.mean(technique_results['clarity_scores'])
                technique_results['avg_response_time'] = np.mean(technique_results['response_times'])
                
                # Calculate total score (weighted average)
                technique_results['total_score'] = (
                    technique_results['avg_faithfulness'] * 0.3 +
                    technique_results['avg_relevance'] * 0.3 +
                    technique_results['avg_completeness'] * 0.25 +
                    technique_results['avg_clarity'] * 0.15
                )
                
                results[technique] = technique_results
                print(f"✅ {technique} completed - Total Score: {technique_results['total_score']:.3f}")
            else:
                print(f"❌ {technique} failed - no valid responses")
        
        return results
    
    def run_comprehensive_evaluation(self, num_queries_per_doc: int = 6) -> Dict:
        """Run evaluation across all documents"""
        documents = self.load_test_documents()
        if not documents:
            print("❌ No test documents found!")
            return {}
        
        print(f"🚀 Starting comprehensive evaluation on {len(documents)} documents")
        print(f"📊 {num_queries_per_doc} queries per document")
        
        all_results = {}
        
        for doc_path in documents:
            doc_name = os.path.basename(doc_path)
            
            try:
                doc_results = self.evaluate_single_document(doc_path, num_queries_per_doc)
                if doc_results:
                    all_results[doc_name] = doc_results
                    print(f"✅ Completed evaluation for {doc_name}")
                else:
                    print(f"❌ Failed to evaluate {doc_name}")
            except Exception as e:
                print(f"❌ Error evaluating {doc_name}: {e}")
                continue
        
        # Calculate overall rankings
        if all_results:
            self.calculate_overall_rankings(all_results)
        
        return all_results
    
    def calculate_overall_rankings(self, all_results: Dict):
        """Calculate and display overall technique rankings"""
        print(f"\n{'='*80}")
        print("📊 OVERALL RAG TECHNIQUE RANKINGS")
        print('='*80)
        
        techniques = ["Adaptive RAG", "CRAG", "Document Augmentation", "Explainable Retrieval", "Basic RAG"]
        overall_scores = {technique: [] for technique in techniques}
        
        # Collect scores across all documents
        for doc_name, doc_results in all_results.items():
            for technique in techniques:
                if technique in doc_results and 'total_score' in doc_results[technique]:
                    overall_scores[technique].append(doc_results[technique]['total_score'])
        
        # Calculate averages and create ranking
        rankings = []
        for technique in techniques:
            if overall_scores[technique]:
                avg_score = np.mean(overall_scores[technique])
                std_score = np.std(overall_scores[technique])
                rankings.append((technique, avg_score, len(overall_scores[technique]), std_score))
        
        # Sort by average score
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🏆 FINAL RANKINGS:")
        print("-" * 70)
        for rank, (technique, avg_score, doc_count, std_score) in enumerate(rankings, 1):
            print(f"{rank}. {technique}: {avg_score:.3f} ± {std_score:.3f} (tested on {doc_count} documents)")
        
        # Detailed breakdown
        print(f"\n📈 DETAILED METRICS BREAKDOWN:")
        print("-" * 80)
        
        metrics = ['avg_faithfulness', 'avg_relevance', 'avg_completeness', 'avg_clarity', 'avg_response_time']
        metric_names = ['Faithfulness', 'Relevance', 'Completeness', 'Clarity', 'Response Time']
        
        for technique, _, _, _ in rankings:
            print(f"\n🔧 {technique}:")
            for metric, name in zip(metrics, metric_names):
                scores = []
                for doc_results in all_results.values():
                    if technique in doc_results and metric in doc_results[technique]:
                        scores.append(doc_results[technique][metric])
                
                if scores:
                    avg_score = np.mean(scores)
                    std_score = np.std(scores)
                    if metric == 'avg_response_time':
                        print(f"   {name}: {avg_score:.2f} ± {std_score:.2f}s")
                    else:
                        print(f"   {name}: {avg_score:.3f} ± {std_score:.3f}")
        
        # Save results
        self.save_results(all_results, rankings)
        
        return rankings
    
    def save_results(self, all_results: Dict, rankings: List):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for doc_name, doc_results in all_results.items():
                json_results[doc_name] = {}
                for technique, tech_results in doc_results.items():
                    json_results[doc_name][technique] = {}
                    for k, v in tech_results.items():
                        if isinstance(v, np.ndarray):
                            json_results[doc_name][technique][k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            json_results[doc_name][technique][k] = float(v)
                        else:
                            json_results[doc_name][technique][k] = v
            json.dump(json_results, f, indent=2)
        
        # Save rankings as CSV
        rankings_df = pd.DataFrame(rankings, columns=['Technique', 'Average_Score', 'Documents_Tested', 'Std_Score'])
        rankings_file = f"technique_rankings_{timestamp}.csv"
        rankings_df.to_csv(rankings_file, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📄 Detailed results: {results_file}")
        print(f"   📊 Rankings: {rankings_file}")

def main():
    """Main execution function"""
    print("🚀 RAG Technique Comprehensive Evaluation")
    print("="*50)
    
    # Check if API key is available
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY not found in environment variables!")
        print("Please set your Google API key before running the evaluation.")
        return
    
    # Initialize evaluator
    evaluator = RAGEvaluator(test_documents_dir="data/")
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(num_queries_per_doc=6)
    
    if results:
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Evaluated {len(results)} documents across multiple RAG techniques")
        print(f"🎯 Results saved with timestamp for future reference")
    else:
        print("❌ Evaluation failed - no results generated")
        print("Please check that test documents exist in the data/ folder")

if __name__ == "__main__":
    main()
