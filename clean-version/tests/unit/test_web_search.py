"""
Unit tests for web search functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.search.web import WebSearcher, DuckDuckGoSearcher, GoogleSearcher
from src.utils.exceptions import SearchError


class TestWebSearcher:
    """Test cases for WebSearcher base class."""
    
    @pytest.fixture
    def web_searcher(self):
        """Create WebSearcher instance for testing."""
        return WebSearcher()
    
    def test_init(self, web_searcher):
        """Test WebSearcher initialization."""
        assert web_searcher.max_results == 5
        assert web_searcher.timeout == 10
    
    def test_search_not_implemented(self, web_searcher):
        """Test that search method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            web_searcher.search("test query")
    
    def test_format_results(self, web_searcher):
        """Test result formatting."""
        raw_results = [
            {"title": "Title 1", "url": "http://example1.com", "snippet": "Snippet 1"},
            {"title": "Title 2", "url": "http://example2.com", "snippet": "Snippet 2"}
        ]
        
        formatted = web_searcher._format_results(raw_results)
        
        assert len(formatted) == 2
        assert formatted[0]["content"] == "Title 1\nSnippet 1"
        assert formatted[0]["metadata"]["source"] == "http://example1.com"
        assert formatted[0]["metadata"]["title"] == "Title 1"
        assert formatted[1]["content"] == "Title 2\nSnippet 2"
        assert formatted[1]["metadata"]["source"] == "http://example2.com"
    
    def test_format_results_empty(self, web_searcher):
        """Test formatting empty results."""
        formatted = web_searcher._format_results([])
        assert formatted == []
    
    def test_format_results_missing_fields(self, web_searcher):
        """Test formatting results with missing fields."""
        raw_results = [
            {"title": "Title 1", "url": "http://example1.com"},  # Missing snippet
            {"title": "Title 2", "snippet": "Snippet 2"}  # Missing URL
        ]
        
        formatted = web_searcher._format_results(raw_results)
        
        assert len(formatted) == 2
        assert formatted[0]["content"] == "Title 1\n"
        assert formatted[1]["metadata"]["source"] == "web_search"  # Default source


class TestDuckDuckGoSearcher:
    """Test cases for DuckDuckGoSearcher class."""
    
    @pytest.fixture
    def ddg_searcher(self):
        """Create DuckDuckGoSearcher instance for testing."""
        return DuckDuckGoSearcher(max_results=3)
    
    def test_init(self, ddg_searcher):
        """Test DuckDuckGoSearcher initialization."""
        assert ddg_searcher.max_results == 3
        assert ddg_searcher.timeout == 10
    
    @patch('duckduckgo_search.DDGS')
    def test_search_success(self, mock_ddgs, ddg_searcher):
        """Test successful DuckDuckGo search."""
        # Mock DDGS response
        mock_ddgs_instance = Mock()
        mock_ddgs.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = [
            {"title": "Result 1", "href": "http://example1.com", "body": "Body 1"},
            {"title": "Result 2", "href": "http://example2.com", "body": "Body 2"}
        ]
        
        query = "test query"
        results = ddg_searcher.search(query)
        
        assert len(results) == 2
        assert results[0]["content"] == "Result 1\nBody 1"
        assert results[0]["metadata"]["source"] == "http://example1.com"
        assert results[0]["metadata"]["title"] == "Result 1"
        assert results[1]["content"] == "Result 2\nBody 2"
        assert results[1]["metadata"]["source"] == "http://example2.com"
        
        mock_ddgs_instance.text.assert_called_once_with(
            keywords=query,
            max_results=3,
            region='us-en',
            safesearch='moderate'
        )
    
    @patch('duckduckgo_search.DDGS')
    def test_search_empty_results(self, mock_ddgs, ddg_searcher):
        """Test DuckDuckGo search with empty results."""
        mock_ddgs_instance = Mock()
        mock_ddgs.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = []
        
        results = ddg_searcher.search("test query")
        
        assert results == []
    
    @patch('duckduckgo_search.DDGS')
    def test_search_exception(self, mock_ddgs, ddg_searcher):
        """Test DuckDuckGo search with exception."""
        mock_ddgs_instance = Mock()
        mock_ddgs.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.side_effect = Exception("Search failed")
        
        with pytest.raises(SearchError):
            ddg_searcher.search("test query")
    
    @patch('duckduckgo_search.DDGS')
    def test_search_with_custom_region(self, mock_ddgs):
        """Test DuckDuckGo search with custom region."""
        searcher = DuckDuckGoSearcher(region='uk-en')
        mock_ddgs_instance = Mock()
        mock_ddgs.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = []
        
        searcher.search("test query")
        
        mock_ddgs_instance.text.assert_called_once_with(
            keywords="test query",
            max_results=5,
            region='uk-en',
            safesearch='moderate'
        )
    
    @patch('duckduckgo_search.DDGS')
    def test_search_with_custom_safesearch(self, mock_ddgs):
        """Test DuckDuckGo search with custom safesearch."""
        searcher = DuckDuckGoSearcher(safesearch='strict')
        mock_ddgs_instance = Mock()
        mock_ddgs.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = []
        
        searcher.search("test query")
        
        mock_ddgs_instance.text.assert_called_once_with(
            keywords="test query",
            max_results=5,
            region='us-en',
            safesearch='strict'
        )


class TestGoogleSearcher:
    """Test cases for GoogleSearcher class."""
    
    @pytest.fixture
    def google_searcher(self):
        """Create GoogleSearcher instance for testing."""
        return GoogleSearcher(
            api_key="test_api_key",
            search_engine_id="test_engine_id",
            max_results=3
        )
    
    def test_init(self, google_searcher):
        """Test GoogleSearcher initialization."""
        assert google_searcher.api_key == "test_api_key"
        assert google_searcher.search_engine_id == "test_engine_id"
        assert google_searcher.max_results == 3
    
    @patch('requests.get')
    def test_search_success(self, mock_get, google_searcher):
        """Test successful Google search."""
        # Mock Google API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "Result 1",
                    "link": "http://example1.com",
                    "snippet": "Snippet 1"
                },
                {
                    "title": "Result 2",
                    "link": "http://example2.com",
                    "snippet": "Snippet 2"
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        query = "test query"
        results = google_searcher.search(query)
        
        assert len(results) == 2
        assert results[0]["content"] == "Result 1\nSnippet 1"
        assert results[0]["metadata"]["source"] == "http://example1.com"
        assert results[0]["metadata"]["title"] == "Result 1"
        assert results[1]["content"] == "Result 2\nSnippet 2"
        assert results[1]["metadata"]["source"] == "http://example2.com"
        
        # Check API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "q=test query" in call_args[1]["params"]["q"]
        assert call_args[1]["params"]["key"] == "test_api_key"
        assert call_args[1]["params"]["cx"] == "test_engine_id"
    
    @patch('requests.get')
    def test_search_no_results(self, mock_get, google_searcher):
        """Test Google search with no results."""
        mock_response = Mock()
        mock_response.json.return_value = {}  # No items
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = google_searcher.search("test query")
        
        assert results == []
    
    @patch('requests.get')
    def test_search_http_error(self, mock_get, google_searcher):
        """Test Google search with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response
        
        with pytest.raises(SearchError):
            google_searcher.search("test query")
    
    @patch('requests.get')
    def test_search_timeout(self, mock_get, google_searcher):
        """Test Google search with timeout."""
        mock_get.side_effect = Exception("Timeout")
        
        with pytest.raises(SearchError):
            google_searcher.search("test query")
    
    @patch('requests.get')
    def test_search_with_language(self, mock_get):
        """Test Google search with custom language."""
        searcher = GoogleSearcher(
            api_key="test_api_key",
            search_engine_id="test_engine_id",
            language="es"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        searcher.search("test query")
        
        call_args = mock_get.call_args
        assert call_args[1]["params"]["lr"] == "lang_es"
    
    @patch('requests.get')
    def test_search_with_date_restrict(self, mock_get):
        """Test Google search with date restriction."""
        searcher = GoogleSearcher(
            api_key="test_api_key",
            search_engine_id="test_engine_id",
            date_restrict="m1"  # Past month
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        searcher.search("test query")
        
        call_args = mock_get.call_args
        assert call_args[1]["params"]["dateRestrict"] == "m1"


class TestSearchUtilities:
    """Test cases for search utility functions."""
    
    def test_get_searcher_duckduckgo(self):
        """Test getting DuckDuckGo searcher."""
        from src.search.web import get_searcher
        
        searcher = get_searcher("duckduckgo")
        
        assert isinstance(searcher, DuckDuckGoSearcher)
    
    def test_get_searcher_google(self):
        """Test getting Google searcher."""
        from src.search.web import get_searcher
        
        searcher = get_searcher(
            "google",
            api_key="test_key",
            search_engine_id="test_id"
        )
        
        assert isinstance(searcher, GoogleSearcher)
    
    def test_get_searcher_invalid(self):
        """Test getting invalid searcher."""
        from src.search.web import get_searcher
        
        with pytest.raises(ValueError):
            get_searcher("invalid_searcher")
    
    def test_search_multiple_sources(self):
        """Test searching multiple sources."""
        from src.search.web import search_multiple_sources
        
        with patch('src.search.web.DuckDuckGoSearcher') as mock_ddg, \
             patch('src.search.web.GoogleSearcher') as mock_google:
            
            # Mock searchers
            mock_ddg_instance = Mock()
            mock_ddg_instance.search.return_value = [
                {"content": "DDG result", "metadata": {"source": "ddg"}}
            ]
            mock_ddg.return_value = mock_ddg_instance
            
            mock_google_instance = Mock()
            mock_google_instance.search.return_value = [
                {"content": "Google result", "metadata": {"source": "google"}}
            ]
            mock_google.return_value = mock_google_instance
            
            searchers = [
                ("duckduckgo", {}),
                ("google", {"api_key": "key", "search_engine_id": "id"})
            ]
            
            results = search_multiple_sources("test query", searchers)
            
            assert len(results) == 2
            assert any("DDG result" in result["content"] for result in results)
            assert any("Google result" in result["content"] for result in results)
    
    def test_deduplicate_search_results(self):
        """Test deduplicating search results."""
        from src.search.web import deduplicate_results
        
        results = [
            {"content": "Result 1", "metadata": {"source": "http://example.com"}},
            {"content": "Result 2", "metadata": {"source": "http://example2.com"}},
            {"content": "Result 1", "metadata": {"source": "http://example.com"}},  # Duplicate
            {"content": "Result 3", "metadata": {"source": "http://example3.com"}}
        ]
        
        deduplicated = deduplicate_results(results)
        
        assert len(deduplicated) == 3
        sources = [result["metadata"]["source"] for result in deduplicated]
        assert len(set(sources)) == 3  # All unique sources
    
    def test_filter_search_results(self):
        """Test filtering search results."""
        from src.search.web import filter_results
        
        results = [
            {"content": "Good result about Python", "metadata": {"source": "example.com"}},
            {"content": "Spam content", "metadata": {"source": "spam.com"}},
            {"content": "Another good result", "metadata": {"source": "good.com"}}
        ]
        
        # Filter by content length
        filtered = filter_results(results, min_content_length=10)
        
        assert len(filtered) == 2
        assert all(len(result["content"]) >= 10 for result in filtered)
    
    def test_rank_search_results(self):
        """Test ranking search results."""
        from src.search.web import rank_results
        
        query = "Python programming"
        results = [
            {"content": "JavaScript tutorial", "metadata": {"source": "js.com"}},
            {"content": "Python programming guide", "metadata": {"source": "python.com"}},
            {"content": "General programming", "metadata": {"source": "general.com"}}
        ]
        
        ranked = rank_results(results, query)
        
        # Python-related result should be ranked higher
        assert "Python programming guide" in ranked[0]["content"]
