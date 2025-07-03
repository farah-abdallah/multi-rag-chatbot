"""
Information retrieval system for documents and web content.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from ..utils.exceptions import RetrievalError
from ..utils.helpers import calculate_similarity, clean_text, chunk_text
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document content."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_id: Optional[str] = None


@dataclass
class RetrievalQuery:
    """Represents a retrieval query."""
    query: str
    max_results: int = 10
    similarity_threshold: float = 0.6
    filter_criteria: Optional[Dict[str, Any]] = None


class DocumentRetriever:
    """Document retrieval system using semantic search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the retrieval system."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing document retriever...")
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_model.get_sentence_embedding_dimension())
            
            self.is_initialized = True
            logger.info("Document retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize document retriever: {str(e)}")
            raise RetrievalError(f"Failed to initialize retriever: {str(e)}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retrieval system."""
        if not self.is_initialized:
            self.initialize()
        
        try:
            logger.info(f"Adding {len(documents)} documents to retriever...")
            
            # Process documents into chunks
            chunks = []
            for doc in documents:
                doc_chunks = self._process_document(doc)
                chunks.extend(doc_chunks)
            
            if not chunks:
                logger.warning("No valid chunks created from documents")
                return
            
            # Generate embeddings
            contents = [chunk.content for chunk in chunks]
            embeddings = self.embedding_model.encode(contents, convert_to_tensor=False)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                chunk.chunk_id = f"chunk_{len(self.documents)}"
                self.documents.append(chunk)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            logger.info(f"Added {len(chunks)} chunks to retrieval system")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise RetrievalError(f"Failed to add documents: {str(e)}")
    
    def _process_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """Process a document into chunks."""
        try:
            content = document.get('content', '')
            if not content:
                return []
            
            # Clean content
            content = clean_text(content)
            
            # Create chunks
            chunks = chunk_text(
                content,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap
            )
            
            # Create DocumentChunk objects
            document_chunks = []
            for i, chunk_content in enumerate(chunks):
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata={
                        **document.get('metadata', {}),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_doc_id': document.get('id', f"doc_{len(self.documents)}")
                    }
                )
                document_chunks.append(chunk)
            
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return []
    
    def retrieve(self, query: RetrievalQuery) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        if not self.is_initialized:
            self.initialize()
        
        if not self.documents:
            logger.warning("No documents in retrieval system")
            return []
        
        try:
            logger.info(f"Retrieving documents for query: {query.query[:100]}...")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query.query], convert_to_tensor=False)
            
            # Search FAISS index
            similarities, indices = self.index.search(
                query_embedding,
                min(query.max_results, len(self.documents))
            )
            
            # Process results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity >= query.similarity_threshold:
                    chunk = self.documents[idx]
                    
                    # Apply filter criteria if provided
                    if query.filter_criteria:
                        if not self._matches_filter(chunk, query.filter_criteria):
                            continue
                    
                    result = {
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'similarity_score': float(similarity),
                        'chunk_id': chunk.chunk_id,
                        'relevance_score': float(similarity)
                    }
                    results.append(result)
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise RetrievalError(f"Retrieval failed: {str(e)}")
    
    def _matches_filter(self, chunk: DocumentChunk, filter_criteria: Dict[str, Any]) -> bool:
        """Check if a chunk matches the filter criteria."""
        for key, value in filter_criteria.items():
            chunk_value = chunk.metadata.get(key)
            if chunk_value != value:
                return False
        return True
    
    def get_similar_chunks(self, chunk_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get chunks similar to a given chunk."""
        if not self.is_initialized:
            self.initialize()
        
        # Find the chunk
        target_chunk = None
        for chunk in self.documents:
            if chunk.chunk_id == chunk_id:
                target_chunk = chunk
                break
        
        if not target_chunk:
            logger.warning(f"Chunk {chunk_id} not found")
            return []
        
        # Use the chunk's embedding for similarity search
        similarities, indices = self.index.search(
            target_chunk.embedding.reshape(1, -1),
            min(max_results + 1, len(self.documents))  # +1 to exclude the chunk itself
        )
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            chunk = self.documents[idx]
            if chunk.chunk_id != chunk_id:  # Exclude the original chunk
                result = {
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'similarity_score': float(similarity),
                    'chunk_id': chunk.chunk_id
                }
                results.append(result)
        
        return results[:max_results]
    
    def clear(self):
        """Clear all documents from the retrieval system."""
        self.documents = []
        if self.index:
            self.index.reset()
        logger.info("Retrieval system cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        return {
            'total_documents': len(self.documents),
            'model_name': self.model_name,
            'is_initialized': self.is_initialized,
            'index_size': self.index.ntotal if self.index else 0
        }


class HybridRetriever:
    """Hybrid retrieval system combining multiple retrieval methods."""
    
    def __init__(self):
        self.document_retriever = DocumentRetriever()
        self.keyword_weights = 0.3
        self.semantic_weights = 0.7
        
    def initialize(self):
        """Initialize the hybrid retriever."""
        self.document_retriever.initialize()
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the hybrid retriever."""
        self.document_retriever.add_documents(documents)
    
    def retrieve(self, query: RetrievalQuery) -> List[Dict[str, Any]]:
        """Retrieve using hybrid approach."""
        # Get semantic results
        semantic_results = self.document_retriever.retrieve(query)
        
        # Get keyword-based results
        keyword_results = self._keyword_retrieve(query)
        
        # Combine and re-rank results
        combined_results = self._combine_results(semantic_results, keyword_results)
        
        return combined_results[:query.max_results]
    
    def _keyword_retrieve(self, query: RetrievalQuery) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval."""
        query_words = set(query.query.lower().split())
        
        results = []
        for chunk in self.document_retriever.documents:
            chunk_words = set(chunk.content.lower().split())
            
            # Calculate keyword similarity
            intersection = query_words.intersection(chunk_words)
            union = query_words.union(chunk_words)
            
            if union:
                similarity = len(intersection) / len(union)
                
                if similarity >= query.similarity_threshold:
                    result = {
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'similarity_score': similarity,
                        'chunk_id': chunk.chunk_id,
                        'method': 'keyword'
                    }
                    results.append(result)
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    def _combine_results(
        self, 
        semantic_results: List[Dict[str, Any]], 
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine semantic and keyword results."""
        # Create a map of chunk_id to results
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result['chunk_id']
            result_map[chunk_id] = result.copy()
            result_map[chunk_id]['semantic_score'] = result['similarity_score']
            result_map[chunk_id]['keyword_score'] = 0.0
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id in result_map:
                result_map[chunk_id]['keyword_score'] = result['similarity_score']
            else:
                result_map[chunk_id] = result.copy()
                result_map[chunk_id]['semantic_score'] = 0.0
                result_map[chunk_id]['keyword_score'] = result['similarity_score']
        
        # Calculate hybrid scores
        for chunk_id, result in result_map.items():
            hybrid_score = (
                result['semantic_score'] * self.semantic_weights +
                result['keyword_score'] * self.keyword_weights
            )
            result['hybrid_score'] = hybrid_score
            result['relevance_score'] = hybrid_score
        
        # Sort by hybrid score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return combined_results
    
    def clear(self):
        """Clear the hybrid retriever."""
        self.document_retriever.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid retriever."""
        stats = self.document_retriever.get_stats()
        stats['retrieval_type'] = 'hybrid'
        stats['keyword_weight'] = self.keyword_weights
        stats['semantic_weight'] = self.semantic_weights
        return stats
