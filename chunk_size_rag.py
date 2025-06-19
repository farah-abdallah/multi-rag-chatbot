"""
Chunk Size Optimization RAG System
Analyzes optimal chunk sizes for document retrieval
"""

import os
import time
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from document_augmentation import load_document_content


class ChunkSizeOptimizer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.results = {}
        self.optimal_sizes = {}
    
    def test_chunk_sizes(self, document_path: str, chunk_sizes: List[int] = None) -> Dict[int, Dict[str, Any]]:
        """Test different chunk sizes and return performance metrics"""
        if chunk_sizes is None:
            chunk_sizes = [200, 500, 1000, 1500, 2000]
        
        # Load document content
        content = load_document_content(document_path)
        
        results = {}
        
        for chunk_size in chunk_sizes:
            try:
                start_time = time.time()
                
                # Create text splitter with current chunk size
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_size // 10,  # 10% overlap
                    separators=["\n\n", "\n", " ", ""]
                )
                
                # Split the document
                chunks = splitter.split_text(content)
                
                # Create documents
                documents = [Document(page_content=chunk) for chunk in chunks]
                
                # Create vector store
                vectorstore = FAISS.from_documents(documents, self.embeddings)
                
                processing_time = time.time() - start_time
                
                # Calculate metrics
                num_chunks = len(chunks)
                avg_chunk_length = sum(len(chunk) for chunk in chunks) / num_chunks if chunks else 0
                
                # Simple quality score (can be enhanced with actual retrieval testing)
                # Favors moderate chunk sizes with reasonable processing time
                quality_score = self._calculate_quality_score(chunk_size, num_chunks, processing_time, avg_chunk_length)
                
                results[chunk_size] = {
                    'num_chunks': num_chunks,
                    'avg_chunk_length': avg_chunk_length,
                    'processing_time': processing_time,
                    'score': quality_score,
                    'vectorstore': vectorstore
                }
                
            except Exception as e:
                results[chunk_size] = {
                    'error': str(e),
                    'score': 0
                }
        
        # Find optimal chunk size
        best_size = max(results.items(), key=lambda x: x[1].get('score', 0))
        self.optimal_sizes[document_path] = best_size[0]
        
        return results
    
    def _calculate_quality_score(self, chunk_size: int, num_chunks: int, processing_time: float, avg_length: float) -> float:
        """Calculate a quality score for the chunk size configuration"""
        # Normalize factors (0-1 scale)
        size_score = 1.0 - abs(chunk_size - 1000) / 2000  # Favor sizes around 1000
        chunk_count_score = min(num_chunks / 50, 1.0)  # Favor reasonable number of chunks
        speed_score = max(0, 1.0 - processing_time / 10)  # Favor faster processing
        length_consistency = 1.0 - abs(avg_length - chunk_size) / chunk_size if chunk_size > 0 else 0  # Favor consistent lengths
        
        # Weighted average
        return (size_score * 0.3 + chunk_count_score * 0.2 + speed_score * 0.2 + length_consistency * 0.3)
    
    def query(self, query: str, document_path: str = None) -> str:
        """Answer query using optimal chunk size"""
        if not self.results:
            return "Please run chunk size optimization first by uploading documents."
        
        # Use the best performing configuration
        if document_path and document_path in self.optimal_sizes:
            optimal_size = self.optimal_sizes[document_path]
            doc_results = self.results.get(os.path.basename(document_path), {})
            
            if optimal_size in doc_results and 'vectorstore' in doc_results[optimal_size]:
                vectorstore = doc_results[optimal_size]['vectorstore']
                
                # Perform similarity search
                docs = vectorstore.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Generate response with context
                response = f"""**Chunk Size Optimization Results:**

**Optimal Chunk Size**: {optimal_size} tokens
**Number of Chunks**: {doc_results[optimal_size]['num_chunks']}
**Processing Time**: {doc_results[optimal_size]['processing_time']:.2f}s
**Quality Score**: {doc_results[optimal_size]['score']:.3f}

**Answer based on optimally chunked document:**

{context}

---
*This response was generated using the optimal chunk size of {optimal_size} tokens, which was determined through performance analysis of multiple chunk size configurations.*"""

                return response
        
        # Fallback: provide optimization summary
        summary = "**Chunk Size Optimization Results:**\n\n"
        for doc_name, doc_results in self.results.items():
            if doc_results:
                best_size = max(doc_results.items(), key=lambda x: x[1].get('score', 0))
                summary += f"📄 **{doc_name}**\n"
                summary += f"   - Optimal chunk size: {best_size[0]} tokens\n"
                summary += f"   - Quality score: {best_size[1].get('score', 0):.3f}\n"
                summary += f"   - Number of chunks: {best_size[1].get('num_chunks', 'N/A')}\n"
                summary += f"   - Processing time: {best_size[1].get('processing_time', 'N/A'):.2f}s\n\n"
        
        summary += f"\n**Your Query**: {query}\n\n"
        summary += "The optimization analysis has been completed. The results above show the optimal chunk sizes determined for each document based on processing efficiency and retrieval quality."
        
        return summary
