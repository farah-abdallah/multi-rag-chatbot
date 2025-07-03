"""
Corrective RAG (CRAG) implementation for enhanced information retrieval.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..llm.gemini import GeminiLLM
from ..search.web import WebSearcher
from ..utils.exceptions import CRAGError
from ..utils.helpers import calculate_similarity
from config.prompts import CRAG_SYSTEM_PROMPT, EVALUATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from information retrieval."""
    content: str
    source: str
    relevance_score: float
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class CRAGResult:
    """Result from CRAG processing."""
    answer: str
    sources: List[RetrievalResult]
    confidence: float
    method_used: str  # 'documents', 'web', 'hybrid'
    additional_info: Dict[str, Any]


class CRAGProcessor:
    """Corrective RAG processor for enhanced information retrieval."""
    
    def __init__(self, llm: GeminiLLM, web_searcher: WebSearcher):
        self.llm = llm
        self.web_searcher = web_searcher
        self.relevance_threshold = 0.6
        self.confidence_threshold = 0.7
        
    async def process_query(
        self, 
        query: str, 
        document_context: List[Dict[str, Any]], 
        use_web_search: bool = True
    ) -> CRAGResult:
        """
        Process a query using CRAG methodology.
        
        Args:
            query: User's question
            document_context: Retrieved document chunks
            use_web_search: Whether to use web search for enhancement
            
        Returns:
            CRAGResult with answer and metadata
        """
        try:
            logger.info(f"Processing CRAG query: {query[:100]}...")
            
            # Step 1: Evaluate document relevance
            doc_evaluation = await self._evaluate_document_relevance(query, document_context)
            
            # Step 2: Determine strategy based on evaluation
            strategy = self._determine_strategy(doc_evaluation)
            
            # Step 3: Execute strategy
            if strategy == 'documents':
                result = await self._process_with_documents(query, document_context)
            elif strategy == 'web':
                result = await self._process_with_web_search(query)
            else:  # hybrid
                result = await self._process_hybrid(query, document_context, use_web_search)
            
            logger.info(f"CRAG processing completed with strategy: {strategy}")
            return result
            
        except Exception as e:
            logger.error(f"Error in CRAG processing: {str(e)}")
            raise CRAGError(f"CRAG processing failed: {str(e)}")
    
    async def _evaluate_document_relevance(
        self, 
        query: str, 
        document_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate the relevance of retrieved documents."""
        if not document_context:
            return {
                'overall_relevance': 0.0,
                'individual_scores': [],
                'recommendation': 'web_search'
            }
        
        individual_scores = []
        
        for doc in document_context:
            # Calculate semantic similarity
            content = doc.get('content', '')
            similarity = calculate_similarity(query, content)
            
            # Use LLM for detailed evaluation
            eval_prompt = EVALUATION_PROMPT.format(
                question=query,
                context=content[:1000]  # Limit context length
            )
            
            try:
                eval_response = await self.llm.generate_response(eval_prompt)
                # Parse evaluation response (simplified)
                relevance_score = self._parse_evaluation_score(eval_response)
                
                individual_scores.append({
                    'content': content,
                    'similarity': similarity,
                    'llm_score': relevance_score,
                    'combined_score': (similarity + relevance_score) / 2,
                    'metadata': doc.get('metadata', {})
                })
                
            except Exception as e:
                logger.warning(f"Error in LLM evaluation: {str(e)}")
                individual_scores.append({
                    'content': content,
                    'similarity': similarity,
                    'llm_score': similarity,
                    'combined_score': similarity,
                    'metadata': doc.get('metadata', {})
                })
        
        # Calculate overall relevance
        if individual_scores:
            overall_relevance = sum(score['combined_score'] for score in individual_scores) / len(individual_scores)
        else:
            overall_relevance = 0.0
        
        # Determine recommendation
        if overall_relevance >= self.confidence_threshold:
            recommendation = 'documents'
        elif overall_relevance >= self.relevance_threshold:
            recommendation = 'hybrid'
        else:
            recommendation = 'web_search'
        
        return {
            'overall_relevance': overall_relevance,
            'individual_scores': individual_scores,
            'recommendation': recommendation
        }
    
    def _determine_strategy(self, evaluation: Dict[str, Any]) -> str:
        """Determine the best strategy based on evaluation."""
        recommendation = evaluation.get('recommendation', 'hybrid')
        overall_relevance = evaluation.get('overall_relevance', 0.0)
        
        # Fine-tune strategy based on relevance score
        if overall_relevance >= 0.8:
            return 'documents'
        elif overall_relevance <= 0.3:
            return 'web'
        else:
            return 'hybrid'
    
    async def _process_with_documents(
        self, 
        query: str, 
        document_context: List[Dict[str, Any]]
    ) -> CRAGResult:
        """Process query using only document context."""
        logger.info("Processing with documents only")
        
        # Prepare context
        context = self._prepare_document_context(document_context)
        
        # Generate response
        prompt = f"{CRAG_SYSTEM_PROMPT}\n\nQuestion: {query}\n\nContext: {context}"
        response = await self.llm.generate_response(prompt)
        
        # Create retrieval results
        sources = [
            RetrievalResult(
                content=doc.get('content', ''),
                source=doc.get('metadata', {}).get('source', 'Unknown'),
                relevance_score=doc.get('relevance_score', 0.0),
                confidence=0.8,
                metadata=doc.get('metadata', {})
            )
            for doc in document_context
        ]
        
        return CRAGResult(
            answer=response,
            sources=sources,
            confidence=0.8,
            method_used='documents',
            additional_info={'context_length': len(context)}
        )
    
    async def _process_with_web_search(self, query: str) -> CRAGResult:
        """Process query using web search."""
        logger.info("Processing with web search only")
        
        # Perform web search
        search_results = await self.web_searcher.search(query)
        
        # Prepare context from search results
        context = self._prepare_web_context(search_results)
        
        # Generate response
        prompt = f"{CRAG_SYSTEM_PROMPT}\n\nQuestion: {query}\n\nWeb Search Results: {context}"
        response = await self.llm.generate_response(prompt)
        
        # Create retrieval results
        sources = [
            RetrievalResult(
                content=result.get('content', ''),
                source=result.get('url', 'Web Search'),
                relevance_score=result.get('relevance', 0.0),
                confidence=0.7,
                metadata=result
            )
            for result in search_results
        ]
        
        return CRAGResult(
            answer=response,
            sources=sources,
            confidence=0.7,
            method_used='web',
            additional_info={'search_results_count': len(search_results)}
        )
    
    async def _process_hybrid(
        self, 
        query: str, 
        document_context: List[Dict[str, Any]],
        use_web_search: bool = True
    ) -> CRAGResult:
        """Process query using hybrid approach."""
        logger.info("Processing with hybrid approach")
        
        # Prepare document context
        doc_context = self._prepare_document_context(document_context)
        
        # Perform web search if enabled
        web_context = ""
        web_results = []
        if use_web_search:
            try:
                web_results = await self.web_searcher.search(query)
                web_context = self._prepare_web_context(web_results)
            except Exception as e:
                logger.warning(f"Web search failed: {str(e)}")
        
        # Combine contexts
        combined_context = f"Document Context:\n{doc_context}\n\nWeb Search Results:\n{web_context}"
        
        # Generate response
        prompt = f"{CRAG_SYSTEM_PROMPT}\n\nQuestion: {query}\n\nCombined Context: {combined_context}"
        response = await self.llm.generate_response(prompt)
        
        # Create retrieval results
        sources = []
        
        # Add document sources
        for doc in document_context:
            sources.append(RetrievalResult(
                content=doc.get('content', ''),
                source=doc.get('metadata', {}).get('source', 'Document'),
                relevance_score=doc.get('relevance_score', 0.0),
                confidence=0.8,
                metadata=doc.get('metadata', {})
            ))
        
        # Add web sources
        for result in web_results:
            sources.append(RetrievalResult(
                content=result.get('content', ''),
                source=result.get('url', 'Web Search'),
                relevance_score=result.get('relevance', 0.0),
                confidence=0.7,
                metadata=result
            ))
        
        return CRAGResult(
            answer=response,
            sources=sources,
            confidence=0.85,
            method_used='hybrid',
            additional_info={
                'doc_context_length': len(doc_context),
                'web_results_count': len(web_results)
            }
        )
    
    def _prepare_document_context(self, document_context: List[Dict[str, Any]]) -> str:
        """Prepare document context for LLM."""
        if not document_context:
            return ""
        
        context_parts = []
        for i, doc in enumerate(document_context):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            source = metadata.get('source', f'Document {i+1}')
            
            context_parts.append(f"[{source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _prepare_web_context(self, web_results: List[Dict[str, Any]]) -> str:
        """Prepare web search context for LLM."""
        if not web_results:
            return ""
        
        context_parts = []
        for result in web_results:
            title = result.get('title', 'Unknown')
            url = result.get('url', '')
            content = result.get('content', '')
            
            context_parts.append(f"[{title}] ({url})\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _parse_evaluation_score(self, evaluation_response: str) -> float:
        """Parse evaluation score from LLM response."""
        # Simple parsing - in a real implementation, this would be more sophisticated
        try:
            # Look for numbers in the response
            import re
            scores = re.findall(r'(\d+(?:\.\d+)?)', evaluation_response)
            if scores:
                # Take the first reasonable score
                score = float(scores[0])
                return min(score / 10.0, 1.0) if score > 1.0 else score
        except:
            pass
        
        # Default to medium relevance
        return 0.5
