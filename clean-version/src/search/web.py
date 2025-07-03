"""
Web search functionality using DuckDuckGo and other search engines.
"""
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode, urlparse
from bs4 import BeautifulSoup
import time
from datetime import datetime

from ..utils.exceptions import SearchError
from ..utils.helpers import clean_text, truncate_text, validate_url
from config.settings import settings

logger = logging.getLogger(__name__)


class WebSearcher:
    """Web search functionality with multiple search engines."""
    
    def __init__(self, max_results: int = 5, timeout: int = 30):
        self.max_results = max_results
        self.timeout = timeout
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def search(self, query: str, engine: str = "duckduckgo") -> List[Dict[str, Any]]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            engine: Search engine to use ('duckduckgo', 'google', 'bing')
            
        Returns:
            List of search results with title, url, content, and metadata
        """
        if not query or not query.strip():
            return []
        
        try:
            logger.info(f"Searching web for: {query[:100]}...")
            
            # Ensure session is available
            if not self.session:
                await self.__aenter__()
            
            if engine == "duckduckgo":
                results = await self._search_duckduckgo(query)
            elif engine == "google":
                results = await self._search_google(query)
            elif engine == "bing":
                results = await self._search_bing(query)
            else:
                logger.warning(f"Unknown search engine: {engine}, using DuckDuckGo")
                results = await self._search_duckduckgo(query)
            
            # Process and clean results
            processed_results = []
            for result in results[:self.max_results]:
                processed_result = await self._process_search_result(result)
                if processed_result:
                    processed_results.append(processed_result)
            
            logger.info(f"Found {len(processed_results)} web search results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            raise SearchError(f"Web search failed: {str(e)}")
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import AsyncDDGS
            
            async with AsyncDDGS() as ddgs:
                results = []
                async for result in ddgs.text(query, max_results=self.max_results):
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'content': result.get('body', ''),
                        'source': 'duckduckgo',
                        'timestamp': datetime.now().isoformat()
                    })
                return results
                
        except ImportError:
            logger.warning("duckduckgo_search not available, using fallback")
            return await self._search_fallback(query)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return await self._search_fallback(query)
    
    async def _search_google(self, query: str) -> List[Dict[str, Any]]:
        """Search using Google (requires API key)."""
        # This would require Google Custom Search API
        # For now, fall back to DuckDuckGo
        logger.warning("Google search not implemented, using DuckDuckGo")
        return await self._search_duckduckgo(query)
    
    async def _search_bing(self, query: str) -> List[Dict[str, Any]]:
        """Search using Bing (requires API key)."""
        # This would require Bing Search API
        # For now, fall back to DuckDuckGo
        logger.warning("Bing search not implemented, using DuckDuckGo")
        return await self._search_duckduckgo(query)
    
    async def _search_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search method using direct HTTP requests."""
        try:
            # Simple fallback using HTTP requests to search engines
            # This is a basic implementation and may not work reliably
            
            search_url = f"https://html.duckduckgo.com/html/?q={urlencode({'q': query})}"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_search_results_html(html, query)
                else:
                    logger.error(f"HTTP error {response.status} in fallback search")
                    return []
                    
        except Exception as e:
            logger.error(f"Fallback search failed: {str(e)}")
            return []
    
    def _parse_search_results_html(self, html: str, query: str) -> List[Dict[str, Any]]:
        """Parse search results from HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # This is a simplified parser - real implementation would be more robust
            for result_div in soup.find_all('div', class_='result'):
                try:
                    title_elem = result_div.find('a', class_='result__a')
                    snippet_elem = result_div.find('a', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        content = snippet_elem.get_text(strip=True)
                        
                        if title and url and content:
                            results.append({
                                'title': title,
                                'url': url,
                                'content': content,
                                'source': 'duckduckgo_fallback',
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            if len(results) >= self.max_results:
                                break
                                
                except Exception as e:
                    logger.debug(f"Error parsing search result: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing search results HTML: {str(e)}")
            return []
    
    async def _process_search_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and enhance a search result."""
        try:
            title = result.get('title', '').strip()
            url = result.get('url', '').strip()
            content = result.get('content', '').strip()
            
            # Validate required fields
            if not title or not url:
                return None
            
            # Validate URL
            if not validate_url(url):
                logger.debug(f"Invalid URL: {url}")
                return None
            
            # Clean and truncate content
            content = clean_text(content)
            content = truncate_text(content, max_length=500)
            
            # Try to extract more content from the page
            enhanced_content = await self._extract_page_content(url)
            if enhanced_content:
                content = enhanced_content
            
            # Calculate relevance score (simplified)
            relevance_score = self._calculate_relevance(title, content)
            
            return {
                'title': title,
                'url': url,
                'content': content,
                'source': result.get('source', 'web'),
                'timestamp': result.get('timestamp', datetime.now().isoformat()),
                'relevance': relevance_score,
                'metadata': {
                    'domain': urlparse(url).netloc,
                    'content_length': len(content),
                    'processed_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing search result: {str(e)}")
            return None
    
    async def _extract_page_content(self, url: str) -> Optional[str]:
        """Extract content from a web page."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Extract text content
                    text = soup.get_text()
                    
                    # Clean and truncate
                    text = clean_text(text)
                    text = truncate_text(text, max_length=1000)
                    
                    return text
                    
        except Exception as e:
            logger.debug(f"Failed to extract content from {url}: {str(e)}")
            return None
    
    def _calculate_relevance(self, title: str, content: str) -> float:
        """Calculate relevance score for a search result."""
        # Simple relevance calculation based on content length and quality
        base_score = 0.5
        
        # Boost for longer content
        if len(content) > 200:
            base_score += 0.2
        
        # Boost for descriptive titles
        if len(title) > 20:
            base_score += 0.1
        
        # Penalize for very short content
        if len(content) < 50:
            base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))
    
    async def search_multiple_engines(self, query: str) -> List[Dict[str, Any]]:
        """Search using multiple engines and combine results."""
        engines = ["duckduckgo"]  # Add more engines as they become available
        
        all_results = []
        
        for engine in engines:
            try:
                results = await self.search(query, engine=engine)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for engine {engine}: {str(e)}")
                continue
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            url = result.get('url', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Sort by relevance
        unique_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        return unique_results[:self.max_results]
    
    async def close(self):
        """Close the session."""
        if self.session:
            await self.session.close()
            self.session = None


# Global web searcher instance
_web_searcher = None


async def get_web_searcher() -> WebSearcher:
    """Get or create a web searcher instance."""
    global _web_searcher
    if _web_searcher is None:
        _web_searcher = WebSearcher(
            max_results=settings.max_search_results,
            timeout=settings.search_timeout
        )
    return _web_searcher
