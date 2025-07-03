"""
Document chunking functionality for splitting documents into manageable pieces.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime

from ..utils.exceptions import DocumentProcessingError
from ..utils.helpers import clean_text, chunk_text, generate_hash
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    start_position: int
    end_position: int
    chunk_size: int
    overlap_size: int
    created_at: str
    source_metadata: Dict[str, Any]


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            'content': self.content,
            'metadata': {
                'chunk_id': self.metadata.chunk_id,
                'document_id': self.metadata.document_id,
                'chunk_index': self.metadata.chunk_index,
                'total_chunks': self.metadata.total_chunks,
                'start_position': self.metadata.start_position,
                'end_position': self.metadata.end_position,
                'chunk_size': self.metadata.chunk_size,
                'overlap_size': self.metadata.overlap_size,
                'created_at': self.metadata.created_at,
                'source_metadata': self.metadata.source_metadata
            }
        }


class DocumentChunker:
    """Document chunking system with multiple strategies."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 200,
        chunking_strategy: str = "sliding_window"
    ):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.chunking_strategy = chunking_strategy
        
        self.strategies = {
            'sliding_window': self._sliding_window_chunk,
            'sentence_boundary': self._sentence_boundary_chunk,
            'paragraph_boundary': self._paragraph_boundary_chunk,
            'semantic_chunk': self._semantic_chunk
        }
        
    def chunk_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: Document dictionary with content and metadata
            
        Returns:
            List of DocumentChunk objects
        """
        try:
            if self.chunking_strategy not in self.strategies:
                raise DocumentProcessingError(f"Unknown chunking strategy: {self.chunking_strategy}")
            
            content = document.get('content', '')
            if not content:
                logger.warning("Empty document content")
                return []
            
            document_id = document.get('id', generate_hash(content))
            
            logger.info(f"Chunking document {document_id} using {self.chunking_strategy} strategy")
            
            # Apply chunking strategy
            chunks = self.strategies[self.chunking_strategy](content, document, document_id)
            
            logger.info(f"Created {len(chunks)} chunks from document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            raise DocumentProcessingError(f"Failed to chunk document: {str(e)}")
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Chunk multiple documents."""
        all_chunks = []
        
        for document in documents:
            try:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking document {document.get('id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _sliding_window_chunk(
        self,
        content: str,
        document: Dict[str, Any],
        document_id: str
    ) -> List[DocumentChunk]:
        """Chunk using sliding window approach."""
        chunks = []
        
        # Use the helper function for basic chunking
        text_chunks = chunk_text(content, self.chunk_size, self.overlap_size)
        
        for i, chunk_content in enumerate(text_chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            
            # Calculate positions
            start_pos = i * (self.chunk_size - self.overlap_size)
            end_pos = start_pos + len(chunk_content)
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=i,
                total_chunks=len(text_chunks),
                start_position=start_pos,
                end_position=end_pos,
                chunk_size=len(chunk_content),
                overlap_size=self.overlap_size,
                created_at=datetime.now().isoformat(),
                source_metadata=document.get('metadata', {})
            )
            
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _sentence_boundary_chunk(
        self,
        content: str,
        document: Dict[str, Any],
        document_id: str
    ) -> List[DocumentChunk]:
        """Chunk at sentence boundaries."""
        chunks = []
        
        # Split into sentences
        sentences = self._split_into_sentences(content)
        
        current_chunk = ""
        current_sentences = []
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = current_chunk.strip()
                if chunk_content:
                    chunk_id = f"{document_id}_chunk_{chunk_index}"
                    
                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will be updated later
                        start_position=0,  # Simplified for sentence boundary
                        end_position=len(chunk_content),
                        chunk_size=len(chunk_content),
                        overlap_size=0,  # No overlap for sentence boundary
                        created_at=datetime.now().isoformat(),
                        source_metadata=document.get('metadata', {})
                    )
                    
                    chunk = DocumentChunk(
                        content=chunk_content,
                        metadata=metadata
                    )
                    
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = sentence
                current_sentences = [sentence]
            else:
                current_chunk += " " + sentence
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                total_chunks=len(chunks) + 1,
                start_position=0,
                end_position=len(current_chunk),
                chunk_size=len(current_chunk),
                overlap_size=0,
                created_at=datetime.now().isoformat(),
                source_metadata=document.get('metadata', {})
            )
            
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        return chunks
    
    def _paragraph_boundary_chunk(
        self,
        content: str,
        document: Dict[str, Any],
        document_id: str
    ) -> List[DocumentChunk]:
        """Chunk at paragraph boundaries."""
        chunks = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = current_chunk.strip()
                if chunk_content:
                    chunk_id = f"{document_id}_chunk_{chunk_index}"
                    
                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will be updated later
                        start_position=0,
                        end_position=len(chunk_content),
                        chunk_size=len(chunk_content),
                        overlap_size=0,
                        created_at=datetime.now().isoformat(),
                        source_metadata=document.get('metadata', {})
                    )
                    
                    chunk = DocumentChunk(
                        content=chunk_content,
                        metadata=metadata
                    )
                    
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                total_chunks=len(chunks) + 1,
                start_position=0,
                end_position=len(current_chunk),
                chunk_size=len(current_chunk),
                overlap_size=0,
                created_at=datetime.now().isoformat(),
                source_metadata=document.get('metadata', {})
            )
            
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        return chunks
    
    def _semantic_chunk(
        self,
        content: str,
        document: Dict[str, Any],
        document_id: str
    ) -> List[DocumentChunk]:
        """Chunk based on semantic similarity (simplified implementation)."""
        # For now, use sentence boundary chunking as a fallback
        # A more sophisticated implementation would use embeddings
        logger.info("Semantic chunking not fully implemented, using sentence boundary")
        return self._sentence_boundary_chunk(content, document, document_id)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved with NLTK or spaCy
        sentence_endings = r'[.!?]+'
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def optimize_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Optimize chunks by merging small chunks and splitting large ones."""
        optimized_chunks = []
        
        for chunk in chunks:
            content = chunk.content
            
            # If chunk is too small, try to merge with next chunk
            if len(content) < self.chunk_size * 0.3:  # Less than 30% of target size
                # For now, keep as is - merging would require more complex logic
                optimized_chunks.append(chunk)
            
            # If chunk is too large, split it
            elif len(content) > self.chunk_size * 1.5:  # More than 150% of target size
                # Split the chunk
                sub_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _split_large_chunk(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """Split a large chunk into smaller ones."""
        content = chunk.content
        target_size = self.chunk_size
        
        # Use sliding window on the large chunk
        text_chunks = chunk_text(content, target_size, self.overlap_size)
        
        sub_chunks = []
        for i, chunk_content in enumerate(text_chunks):
            chunk_id = f"{chunk.metadata.chunk_id}_sub_{i}"
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=chunk.metadata.document_id,
                chunk_index=chunk.metadata.chunk_index,
                total_chunks=len(text_chunks),
                start_position=i * (target_size - self.overlap_size),
                end_position=i * (target_size - self.overlap_size) + len(chunk_content),
                chunk_size=len(chunk_content),
                overlap_size=self.overlap_size,
                created_at=datetime.now().isoformat(),
                source_metadata=chunk.metadata.source_metadata
            )
            
            sub_chunk = DocumentChunk(
                content=chunk_content,
                metadata=metadata
            )
            
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking process."""
        if not chunks:
            return {
                'total_chunks': 0,
                'average_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_content_length': 0
            }
        
        chunk_sizes = [chunk.metadata.chunk_size for chunk in chunks]
        total_content_length = sum(chunk_sizes)
        
        return {
            'total_chunks': len(chunks),
            'average_chunk_size': total_content_length / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_content_length': total_content_length,
            'chunking_strategy': self.chunking_strategy,
            'target_chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size
        }
