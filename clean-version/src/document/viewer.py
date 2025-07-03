"""
Document viewing and source highlighting functionality.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime

from ..utils.exceptions import DocumentProcessingError
from ..utils.helpers import clean_text, truncate_text

logger = logging.getLogger(__name__)


@dataclass
class SourceHighlight:
    """Represents a highlighted source in a document."""
    document_id: str
    document_name: str
    chunk_id: str
    highlighted_text: str
    context_before: str
    context_after: str
    relevance_score: float
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = None


class DocumentViewer:
    """Document viewer with source highlighting capabilities."""
    
    def __init__(self, context_length: int = 200):
        self.context_length = context_length
        self.documents = {}  # document_id -> document
        self.chunks = {}     # chunk_id -> chunk
        
    def add_document(self, document: Dict[str, Any]):
        """Add a document to the viewer."""
        document_id = document.get('id', 'unknown')
        self.documents[document_id] = document
        logger.debug(f"Added document {document_id} to viewer")
    
    def add_chunk(self, chunk: Dict[str, Any]):
        """Add a chunk to the viewer."""
        chunk_id = chunk.get('metadata', {}).get('chunk_id', 'unknown')
        self.chunks[chunk_id] = chunk
        logger.debug(f"Added chunk {chunk_id} to viewer")
    
    def highlight_sources(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        answer: str
    ) -> List[SourceHighlight]:
        """
        Highlight sources in documents based on retrieved chunks and answer.
        
        Args:
            query: Original query
            retrieved_chunks: List of retrieved chunks
            answer: Generated answer
            
        Returns:
            List of SourceHighlight objects
        """
        try:
            logger.info(f"Highlighting sources for query: {query[:100]}...")
            
            highlights = []
            
            for chunk in retrieved_chunks:
                chunk_id = chunk.get('metadata', {}).get('chunk_id', '')
                document_id = chunk.get('metadata', {}).get('document_id', '')
                
                if not chunk_id or not document_id:
                    continue
                
                # Find relevant text in the chunk
                relevant_text = self._find_relevant_text(query, chunk['content'], answer)
                
                if relevant_text:
                    highlight = self._create_highlight(
                        chunk, document_id, chunk_id, relevant_text, query, answer
                    )
                    highlights.append(highlight)
            
            # Sort by relevance score
            highlights.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"Created {len(highlights)} source highlights")
            return highlights
            
        except Exception as e:
            logger.error(f"Error highlighting sources: {str(e)}")
            raise DocumentProcessingError(f"Failed to highlight sources: {str(e)}")
    
    def _find_relevant_text(self, query: str, content: str, answer: str) -> Optional[str]:
        """Find the most relevant text in a chunk."""
        # Simple approach: find text that appears in both chunk and answer
        answer_words = set(answer.lower().split())
        query_words = set(query.lower().split())
        
        # Split content into sentences
        sentences = self._split_into_sentences(content)
        
        best_sentence = None
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            
            # Calculate overlap with answer and query
            answer_overlap = len(sentence_words.intersection(answer_words))
            query_overlap = len(sentence_words.intersection(query_words))
            
            # Combined score
            score = answer_overlap * 2 + query_overlap
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return best_sentence if best_score > 0 else None
    
    def _create_highlight(
        self,
        chunk: Dict[str, Any],
        document_id: str,
        chunk_id: str,
        relevant_text: str,
        query: str,
        answer: str
    ) -> SourceHighlight:
        """Create a source highlight object."""
        content = chunk['content']
        metadata = chunk.get('metadata', {})
        
        # Find position of relevant text in chunk
        text_position = content.find(relevant_text)
        
        # Extract context
        context_before = ""
        context_after = ""
        
        if text_position >= 0:
            context_start = max(0, text_position - self.context_length)
            context_end = min(len(content), text_position + len(relevant_text) + self.context_length)
            
            context_before = content[context_start:text_position]
            context_after = content[text_position + len(relevant_text):context_end]
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(relevant_text, query, answer)
        
        # Get document name
        document = self.documents.get(document_id, {})
        document_name = document.get('metadata', {}).get('filename', 'Unknown Document')
        
        return SourceHighlight(
            document_id=document_id,
            document_name=document_name,
            chunk_id=chunk_id,
            highlighted_text=relevant_text,
            context_before=context_before,
            context_after=context_after,
            relevance_score=relevance_score,
            page_number=metadata.get('page_number'),
            section=metadata.get('section'),
            metadata=metadata
        )
    
    def _calculate_relevance_score(self, text: str, query: str, answer: str) -> float:
        """Calculate relevance score for highlighted text."""
        text_words = set(text.lower().split())
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Calculate overlaps
        query_overlap = len(text_words.intersection(query_words))
        answer_overlap = len(text_words.intersection(answer_words))
        
        # Normalize by text length
        total_words = len(text_words)
        if total_words == 0:
            return 0.0
        
        query_score = query_overlap / total_words
        answer_score = answer_overlap / total_words
        
        # Combined score (weighted toward answer relevance)
        combined_score = (query_score * 0.3 + answer_score * 0.7)
        
        return min(1.0, combined_score)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentence_endings = r'[.!?]+'
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def get_document_preview(self, document_id: str, max_length: int = 500) -> str:
        """Get a preview of a document."""
        document = self.documents.get(document_id)
        if not document:
            return "Document not found"
        
        content = document.get('content', '')
        return truncate_text(content, max_length)
    
    def get_chunk_context(self, chunk_id: str, extended_context: bool = False) -> Dict[str, Any]:
        """Get context for a specific chunk."""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return {'error': 'Chunk not found'}
        
        metadata = chunk.get('metadata', {})
        document_id = metadata.get('document_id', '')
        
        result = {
            'chunk_id': chunk_id,
            'document_id': document_id,
            'content': chunk['content'],
            'metadata': metadata
        }
        
        if extended_context:
            # Get neighboring chunks
            chunk_index = metadata.get('chunk_index', 0)
            total_chunks = metadata.get('total_chunks', 1)
            
            # Find previous and next chunks
            prev_chunk = None
            next_chunk = None
            
            for cid, c in self.chunks.items():
                c_metadata = c.get('metadata', {})
                if c_metadata.get('document_id') == document_id:
                    c_index = c_metadata.get('chunk_index', 0)
                    if c_index == chunk_index - 1:
                        prev_chunk = c
                    elif c_index == chunk_index + 1:
                        next_chunk = c
            
            result['previous_chunk'] = prev_chunk['content'] if prev_chunk else None
            result['next_chunk'] = next_chunk['content'] if next_chunk else None
            result['chunk_position'] = f"{chunk_index + 1} of {total_chunks}"
        
        return result
    
    def search_in_documents(self, search_term: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for a term in all documents."""
        results = []
        
        for document_id, document in self.documents.items():
            content = document.get('content', '').lower()
            search_term_lower = search_term.lower()
            
            if search_term_lower in content:
                # Find all occurrences
                start = 0
                while True:
                    pos = content.find(search_term_lower, start)
                    if pos == -1:
                        break
                    
                    # Extract context around the match
                    context_start = max(0, pos - self.context_length)
                    context_end = min(len(content), pos + len(search_term) + self.context_length)
                    
                    context = content[context_start:context_end]
                    
                    result = {
                        'document_id': document_id,
                        'document_name': document.get('metadata', {}).get('filename', 'Unknown'),
                        'position': pos,
                        'context': context,
                        'highlighted_term': search_term
                    }
                    
                    results.append(result)
                    
                    if len(results) >= max_results:
                        break
                    
                    start = pos + len(search_term)
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def export_highlights(self, highlights: List[SourceHighlight], format: str = 'json') -> str:
        """Export highlights in various formats."""
        if format == 'json':
            import json
            
            export_data = []
            for highlight in highlights:
                export_data.append({
                    'document_id': highlight.document_id,
                    'document_name': highlight.document_name,
                    'chunk_id': highlight.chunk_id,
                    'highlighted_text': highlight.highlighted_text,
                    'context_before': highlight.context_before,
                    'context_after': highlight.context_after,
                    'relevance_score': highlight.relevance_score,
                    'page_number': highlight.page_number,
                    'section': highlight.section,
                    'metadata': highlight.metadata
                })
            
            return json.dumps(export_data, indent=2)
        
        elif format == 'text':
            lines = []
            for i, highlight in enumerate(highlights, 1):
                lines.append(f"=== Highlight {i} ===")
                lines.append(f"Document: {highlight.document_name}")
                lines.append(f"Relevance: {highlight.relevance_score:.2f}")
                lines.append(f"Text: {highlight.highlighted_text}")
                lines.append(f"Context: ...{highlight.context_before}{highlight.highlighted_text}{highlight.context_after}...")
                lines.append("")
            
            return '\n'.join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_viewer_stats(self) -> Dict[str, Any]:
        """Get statistics about the document viewer."""
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'context_length': self.context_length,
            'documents': [
                {
                    'document_id': doc_id,
                    'filename': doc.get('metadata', {}).get('filename', 'Unknown'),
                    'content_length': len(doc.get('content', ''))
                }
                for doc_id, doc in self.documents.items()
            ]
        }
