"""
Knowledge refinement and management system.
"""
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from ..llm.gemini import GeminiLLM
from ..utils.exceptions import KnowledgeError
from ..utils.helpers import generate_hash, clean_text, extract_keywords
from config.prompts import KNOWLEDGE_REFINEMENT_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """Represents a knowledge item."""
    id: str
    content: str
    keywords: List[str]
    category: str
    confidence: float
    sources: List[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class KnowledgeQuery:
    """Represents a knowledge query."""
    query: str
    category: Optional[str] = None
    min_confidence: float = 0.5
    max_results: int = 10


class KnowledgeBase:
    """Knowledge base for storing and retrieving refined knowledge."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("data/knowledge_base.json")
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.categories: Set[str] = set()
        self.load_knowledge()
        
    def add_knowledge_item(self, item: KnowledgeItem) -> None:
        """Add a knowledge item to the base."""
        self.knowledge_items[item.id] = item
        self.categories.add(item.category)
        logger.info(f"Added knowledge item: {item.id}")
        
    def get_knowledge_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item by ID."""
        return self.knowledge_items.get(item_id)
        
    def search_knowledge(self, query: KnowledgeQuery) -> List[KnowledgeItem]:
        """Search for knowledge items."""
        results = []
        query_keywords = extract_keywords(query.query)
        
        for item in self.knowledge_items.values():
            # Filter by category if specified
            if query.category and item.category != query.category:
                continue
                
            # Filter by confidence
            if item.confidence < query.min_confidence:
                continue
                
            # Calculate relevance score
            relevance = self._calculate_relevance(query_keywords, item)
            
            if relevance > 0.1:  # Minimum relevance threshold
                results.append((item, relevance))
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:query.max_results]]
    
    def _calculate_relevance(self, query_keywords: List[str], item: KnowledgeItem) -> float:
        """Calculate relevance score between query and knowledge item."""
        if not query_keywords:
            return 0.0
            
        # Keyword matching
        matching_keywords = set(query_keywords).intersection(set(item.keywords))
        keyword_score = len(matching_keywords) / len(query_keywords)
        
        # Content similarity (simplified)
        content_score = 0.0
        for keyword in query_keywords:
            if keyword.lower() in item.content.lower():
                content_score += 1
        content_score = content_score / len(query_keywords)
        
        # Combine scores
        relevance = (keyword_score * 0.6 + content_score * 0.4)
        
        # Boost by confidence
        relevance *= item.confidence
        
        return relevance
    
    def update_knowledge_item(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update a knowledge item."""
        if item_id not in self.knowledge_items:
            return False
            
        item = self.knowledge_items[item_id]
        
        # Update fields
        if 'content' in updates:
            item.content = updates['content']
        if 'keywords' in updates:
            item.keywords = updates['keywords']
        if 'category' in updates:
            self.categories.discard(item.category)
            item.category = updates['category']
            self.categories.add(item.category)
        if 'confidence' in updates:
            item.confidence = updates['confidence']
        if 'sources' in updates:
            item.sources = updates['sources']
        if 'metadata' in updates:
            item.metadata.update(updates['metadata'])
            
        item.updated_at = datetime.now()
        
        logger.info(f"Updated knowledge item: {item_id}")
        return True
    
    def delete_knowledge_item(self, item_id: str) -> bool:
        """Delete a knowledge item."""
        if item_id in self.knowledge_items:
            del self.knowledge_items[item_id]
            logger.info(f"Deleted knowledge item: {item_id}")
            return True
        return False
    
    def save_knowledge(self) -> None:
        """Save knowledge base to storage."""
        try:
            # Create directory if it doesn't exist
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            data = {
                'items': {},
                'categories': list(self.categories),
                'saved_at': datetime.now().isoformat()
            }
            
            for item_id, item in self.knowledge_items.items():
                data['items'][item_id] = {
                    'id': item.id,
                    'content': item.content,
                    'keywords': item.keywords,
                    'category': item.category,
                    'confidence': item.confidence,
                    'sources': item.sources,
                    'created_at': item.created_at.isoformat(),
                    'updated_at': item.updated_at.isoformat(),
                    'metadata': item.metadata
                }
            
            # Save to file
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved knowledge base to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {str(e)}")
            raise KnowledgeError(f"Failed to save knowledge base: {str(e)}")
    
    def load_knowledge(self) -> None:
        """Load knowledge base from storage."""
        try:
            if not self.storage_path.exists():
                logger.info("No existing knowledge base found")
                return
                
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load items
            for item_id, item_data in data.get('items', {}).items():
                item = KnowledgeItem(
                    id=item_data['id'],
                    content=item_data['content'],
                    keywords=item_data['keywords'],
                    category=item_data['category'],
                    confidence=item_data['confidence'],
                    sources=item_data['sources'],
                    created_at=datetime.fromisoformat(item_data['created_at']),
                    updated_at=datetime.fromisoformat(item_data['updated_at']),
                    metadata=item_data['metadata']
                )
                self.knowledge_items[item_id] = item
            
            # Load categories
            self.categories = set(data.get('categories', []))
            
            logger.info(f"Loaded {len(self.knowledge_items)} knowledge items")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            # Don't raise error - continue with empty knowledge base
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            'total_items': len(self.knowledge_items),
            'categories': list(self.categories),
            'average_confidence': sum(item.confidence for item in self.knowledge_items.values()) / len(self.knowledge_items) if self.knowledge_items else 0,
            'storage_path': str(self.storage_path),
            'last_updated': max(item.updated_at for item in self.knowledge_items.values()).isoformat() if self.knowledge_items else None
        }


class KnowledgeRefinementSystem:
    """System for refining and managing knowledge."""
    
    def __init__(self, llm: GeminiLLM, knowledge_base: Optional[KnowledgeBase] = None):
        self.llm = llm
        self.knowledge_base = knowledge_base or KnowledgeBase()
        
    async def refine_knowledge(
        self,
        query: str,
        context: List[Dict[str, Any]],
        existing_knowledge: Optional[List[KnowledgeItem]] = None
    ) -> List[KnowledgeItem]:
        """
        Refine knowledge from context and existing knowledge.
        
        Args:
            query: The original query
            context: List of context documents
            existing_knowledge: Optional existing knowledge items
            
        Returns:
            List of refined knowledge items
        """
        try:
            logger.info(f"Refining knowledge for query: {query[:100]}...")
            
            # Prepare context
            combined_context = self._prepare_context(context)
            
            # Generate refinement prompt
            prompt = KNOWLEDGE_REFINEMENT_PROMPT.format(
                question=query,
                context=combined_context
            )
            
            # Get refined knowledge from LLM
            refined_response = await self.llm.generate_response(prompt)
            
            # Extract knowledge items from response
            knowledge_items = await self._extract_knowledge_items(
                refined_response, query, context
            )
            
            # Add to knowledge base
            for item in knowledge_items:
                self.knowledge_base.add_knowledge_item(item)
            
            # Save knowledge base
            self.knowledge_base.save_knowledge()
            
            logger.info(f"Refined {len(knowledge_items)} knowledge items")
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Knowledge refinement failed: {str(e)}")
            raise KnowledgeError(f"Knowledge refinement failed: {str(e)}")
    
    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Prepare context for knowledge refinement."""
        context_parts = []
        
        for i, doc in enumerate(context):
            content = doc.get('content', '')
            source = doc.get('metadata', {}).get('source', f'Document {i+1}')
            
            context_parts.append(f"[{source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    async def _extract_knowledge_items(
        self,
        refined_response: str,
        query: str,
        context: List[Dict[str, Any]]
    ) -> List[KnowledgeItem]:
        """Extract knowledge items from refined response."""
        try:
            # For now, create a single knowledge item from the response
            # In a more sophisticated implementation, this would parse multiple items
            
            content = clean_text(refined_response)
            keywords = extract_keywords(content)
            
            # Determine category based on query
            category = await self._determine_category(query, content)
            
            # Calculate confidence based on context quality
            confidence = self._calculate_confidence(context)
            
            # Extract sources
            sources = []
            for doc in context:
                source = doc.get('metadata', {}).get('source', '')
                if source:
                    sources.append(source)
            
            # Create knowledge item
            item_id = generate_hash(f"{query}:{content}")
            
            item = KnowledgeItem(
                id=item_id,
                content=content,
                keywords=keywords,
                category=category,
                confidence=confidence,
                sources=sources,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={
                    'original_query': query,
                    'context_count': len(context),
                    'refinement_method': 'llm_refinement'
                }
            )
            
            return [item]
            
        except Exception as e:
            logger.error(f"Failed to extract knowledge items: {str(e)}")
            return []
    
    async def _determine_category(self, query: str, content: str) -> str:
        """Determine category for knowledge item."""
        try:
            prompt = f"""
            Determine the most appropriate category for this knowledge item.
            Choose from: general, technical, business, science, health, education, entertainment, other
            
            Query: {query}
            Content: {content[:500]}
            
            Respond with just the category name:
            """
            
            response = await self.llm.generate_response(prompt)
            category = response.strip().lower()
            
            # Validate category
            valid_categories = {
                'general', 'technical', 'business', 'science', 
                'health', 'education', 'entertainment', 'other'
            }
            
            return category if category in valid_categories else 'general'
            
        except Exception as e:
            logger.warning(f"Category determination failed: {str(e)}")
            return 'general'
    
    def _calculate_confidence(self, context: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for knowledge item."""
        if not context:
            return 0.5
        
        # Base confidence
        confidence = 0.5
        
        # Boost for more sources
        confidence += min(0.3, len(context) * 0.1)
        
        # Boost for longer content
        total_content_length = sum(len(doc.get('content', '')) for doc in context)
        if total_content_length > 1000:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    async def enhance_knowledge(self, item_id: str, additional_context: List[Dict[str, Any]]) -> bool:
        """Enhance existing knowledge item with additional context."""
        try:
            item = self.knowledge_base.get_knowledge_item(item_id)
            if not item:
                logger.warning(f"Knowledge item {item_id} not found")
                return False
            
            # Prepare additional context
            additional_content = self._prepare_context(additional_context)
            
            # Generate enhancement prompt
            prompt = f"""
            Enhance the following knowledge item with additional context:
            
            Existing Knowledge: {item.content}
            Additional Context: {additional_content}
            
            Provide an enhanced version that incorporates the new information:
            """
            
            enhanced_response = await self.llm.generate_response(prompt)
            
            # Update knowledge item
            updates = {
                'content': enhanced_response,
                'keywords': extract_keywords(enhanced_response),
                'sources': item.sources + [doc.get('metadata', {}).get('source', '') for doc in additional_context],
                'confidence': min(1.0, item.confidence + 0.1),  # Slight confidence boost
                'metadata': {
                    **item.metadata,
                    'enhanced_at': datetime.now().isoformat(),
                    'enhancement_count': item.metadata.get('enhancement_count', 0) + 1
                }
            }
            
            success = self.knowledge_base.update_knowledge_item(item_id, updates)
            
            if success:
                self.knowledge_base.save_knowledge()
                logger.info(f"Enhanced knowledge item: {item_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Knowledge enhancement failed: {str(e)}")
            return False
    
    async def get_relevant_knowledge(self, query: str, max_results: int = 5) -> List[KnowledgeItem]:
        """Get relevant knowledge items for a query."""
        knowledge_query = KnowledgeQuery(
            query=query,
            max_results=max_results
        )
        
        return self.knowledge_base.search_knowledge(knowledge_query)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge refinement system statistics."""
        return {
            'knowledge_base_stats': self.knowledge_base.get_stats(),
            'refinement_system': {
                'llm_model': self.llm.model_name,
                'initialized': True
            }
        }
