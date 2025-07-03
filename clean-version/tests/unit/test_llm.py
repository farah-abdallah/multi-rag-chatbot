"""
Unit tests for LLM functionality including Gemini and API management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm.gemini import GeminiLLM
from src.llm.api_manager import APIKeyManager
from src.utils.exceptions import LLMError, APIKeyError


class TestGeminiLLM:
    """Test cases for GeminiLLM class."""
    
    @pytest.fixture
    def mock_genai(self):
        """Create mock Google Generative AI for testing."""
        with patch('src.llm.gemini.genai') as mock_genai:
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            yield mock_genai, mock_model
    
    @pytest.fixture
    def gemini_llm(self, mock_genai):
        """Create GeminiLLM instance for testing."""
        mock_genai_module, mock_model = mock_genai
        return GeminiLLM(api_key="test_api_key", model_name="gemini-pro")
    
    def test_init(self, gemini_llm, mock_genai):
        """Test GeminiLLM initialization."""
        mock_genai_module, mock_model = mock_genai
        
        assert gemini_llm.api_key == "test_api_key"
        assert gemini_llm.model_name == "gemini-pro"
        assert gemini_llm.model is not None
        mock_genai_module.configure.assert_called_once_with(api_key="test_api_key")
    
    def test_generate_success(self, gemini_llm, mock_genai):
        """Test successful text generation."""
        mock_genai_module, mock_model = mock_genai
        
        # Mock response
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_model.generate_content.return_value = mock_response
        
        prompt = "Test prompt"
        response = gemini_llm.generate(prompt)
        
        assert response == "Generated response"
        mock_model.generate_content.assert_called_once_with(prompt)
    
    def test_generate_with_context(self, gemini_llm, mock_genai):
        """Test text generation with context."""
        mock_genai_module, mock_model = mock_genai
        
        mock_response = Mock()
        mock_response.text = "Generated response with context"
        mock_model.generate_content.return_value = mock_response
        
        prompt = "Test prompt"
        context = "Test context"
        response = gemini_llm.generate(prompt, context=context)
        
        assert response == "Generated response with context"
        # Check that prompt was combined with context
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Test context" in call_args
        assert "Test prompt" in call_args
    
    def test_generate_with_sources(self, gemini_llm, mock_genai):
        """Test text generation with sources."""
        mock_genai_module, mock_model = mock_genai
        
        mock_response = Mock()
        mock_response.text = "Generated response with sources"
        mock_model.generate_content.return_value = mock_response
        
        prompt = "Test prompt"
        sources = [
            {"content": "Source 1", "metadata": {"source": "doc1.pdf"}},
            {"content": "Source 2", "metadata": {"source": "doc2.pdf"}}
        ]
        
        response = gemini_llm.generate(prompt, sources=sources)
        
        assert response == "Generated response with sources"
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Source 1" in call_args
        assert "Source 2" in call_args
    
    def test_generate_with_temperature(self, gemini_llm, mock_genai):
        """Test text generation with custom temperature."""
        mock_genai_module, mock_model = mock_genai
        
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_model.generate_content.return_value = mock_response
        
        prompt = "Test prompt"
        response = gemini_llm.generate(prompt, temperature=0.8)
        
        assert response == "Generated response"
        # Check that generation_config was set
        call_args = mock_model.generate_content.call_args
        assert 'generation_config' in call_args.kwargs
    
    def test_generate_api_error(self, gemini_llm, mock_genai):
        """Test generation with API error."""
        mock_genai_module, mock_model = mock_genai
        
        mock_model.generate_content.side_effect = Exception("API Error")
        
        with pytest.raises(LLMError):
            gemini_llm.generate("Test prompt")
    
    def test_generate_empty_response(self, gemini_llm, mock_genai):
        """Test generation with empty response."""
        mock_genai_module, mock_model = mock_genai
        
        mock_response = Mock()
        mock_response.text = ""
        mock_model.generate_content.return_value = mock_response
        
        with pytest.raises(LLMError):
            gemini_llm.generate("Test prompt")
    
    def test_generate_batch(self, gemini_llm, mock_genai):
        """Test batch text generation."""
        mock_genai_module, mock_model = mock_genai
        
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_model.generate_content.return_value = mock_response
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = gemini_llm.generate_batch(prompts)
        
        assert len(responses) == 3
        assert all(response == "Generated response" for response in responses)
        assert mock_model.generate_content.call_count == 3
    
    def test_count_tokens(self, gemini_llm, mock_genai):
        """Test token counting."""
        mock_genai_module, mock_model = mock_genai
        
        mock_model.count_tokens.return_value = Mock(total_tokens=10)
        
        text = "Test text"
        token_count = gemini_llm.count_tokens(text)
        
        assert token_count == 10
        mock_model.count_tokens.assert_called_once_with(text)
    
    def test_format_prompt_simple(self, gemini_llm):
        """Test simple prompt formatting."""
        prompt = "Test prompt"
        formatted = gemini_llm._format_prompt(prompt)
        
        assert formatted == prompt
    
    def test_format_prompt_with_context_and_sources(self, gemini_llm):
        """Test prompt formatting with context and sources."""
        prompt = "Test prompt"
        context = "Test context"
        sources = [
            {"content": "Source 1", "metadata": {"source": "doc1.pdf"}},
            {"content": "Source 2", "metadata": {"source": "doc2.pdf"}}
        ]
        
        formatted = gemini_llm._format_prompt(prompt, context=context, sources=sources)
        
        assert "Test context" in formatted
        assert "Test prompt" in formatted
        assert "Source 1" in formatted
        assert "Source 2" in formatted
        assert "doc1.pdf" in formatted
        assert "doc2.pdf" in formatted


class TestAPIKeyManager:
    """Test cases for APIKeyManager class."""
    
    @pytest.fixture
    def api_keys(self):
        """Create sample API keys for testing."""
        return ["key1", "key2", "key3"]
    
    @pytest.fixture
    def api_manager(self, api_keys):
        """Create APIKeyManager instance for testing."""
        return APIKeyManager(api_keys)
    
    def test_init(self, api_manager, api_keys):
        """Test APIKeyManager initialization."""
        assert api_manager.api_keys == api_keys
        assert api_manager.current_key_index == 0
        assert api_manager.key_stats == {}
    
    def test_get_current_key(self, api_manager):
        """Test getting current API key."""
        current_key = api_manager.get_current_key()
        
        assert current_key == "key1"
        assert api_manager.key_stats["key1"]["requests"] == 1
        assert api_manager.key_stats["key1"]["successes"] == 0
        assert api_manager.key_stats["key1"]["failures"] == 0
    
    def test_rotate_key(self, api_manager):
        """Test API key rotation."""
        initial_key = api_manager.get_current_key()
        api_manager.rotate_key()
        next_key = api_manager.get_current_key()
        
        assert next_key != initial_key
        assert next_key == "key2"
        assert api_manager.current_key_index == 1
    
    def test_rotate_key_wrap_around(self, api_manager):
        """Test API key rotation wrapping around."""
        # Rotate to last key
        api_manager.current_key_index = 2
        api_manager.rotate_key()
        
        # Should wrap around to first key
        assert api_manager.current_key_index == 0
        assert api_manager.get_current_key() == "key1"
    
    def test_mark_key_success(self, api_manager):
        """Test marking key as successful."""
        key = api_manager.get_current_key()
        api_manager.mark_key_success(key)
        
        assert api_manager.key_stats[key]["successes"] == 1
        assert api_manager.key_stats[key]["failures"] == 0
    
    def test_mark_key_failure(self, api_manager):
        """Test marking key as failed."""
        key = api_manager.get_current_key()
        api_manager.mark_key_failure(key)
        
        assert api_manager.key_stats[key]["failures"] == 1
        assert api_manager.key_stats[key]["successes"] == 0
    
    def test_get_key_stats(self, api_manager):
        """Test getting key statistics."""
        key = api_manager.get_current_key()
        api_manager.mark_key_success(key)
        api_manager.mark_key_failure(key)
        
        stats = api_manager.get_key_stats(key)
        
        assert stats["requests"] == 1
        assert stats["successes"] == 1
        assert stats["failures"] == 1
        assert stats["success_rate"] == 0.5
    
    def test_get_all_stats(self, api_manager):
        """Test getting all key statistics."""
        # Use multiple keys
        key1 = api_manager.get_current_key()
        api_manager.mark_key_success(key1)
        
        api_manager.rotate_key()
        key2 = api_manager.get_current_key()
        api_manager.mark_key_failure(key2)
        
        all_stats = api_manager.get_all_stats()
        
        assert key1 in all_stats
        assert key2 in all_stats
        assert all_stats[key1]["successes"] == 1
        assert all_stats[key2]["failures"] == 1
    
    def test_get_best_key(self, api_manager):
        """Test getting best performing key."""
        # Set up different success rates
        api_manager.get_current_key()  # key1
        api_manager.mark_key_success("key1")
        api_manager.mark_key_success("key1")
        
        api_manager.rotate_key()
        api_manager.get_current_key()  # key2
        api_manager.mark_key_success("key2")
        api_manager.mark_key_failure("key2")
        
        best_key = api_manager.get_best_key()
        
        assert best_key == "key1"  # 100% success rate vs 50%
    
    def test_is_key_healthy(self, api_manager):
        """Test checking if key is healthy."""
        key = api_manager.get_current_key()
        
        # Initially healthy (no failures)
        assert api_manager.is_key_healthy(key)
        
        # Mark some failures
        api_manager.mark_key_failure(key)
        api_manager.mark_key_failure(key)
        api_manager.mark_key_failure(key)
        
        # Should still be healthy if failure rate is not too high
        assert api_manager.is_key_healthy(key, failure_threshold=0.8)
        
        # Should be unhealthy with lower threshold
        assert not api_manager.is_key_healthy(key, failure_threshold=0.5)
    
    def test_auto_rotate_on_failure(self, api_manager):
        """Test automatic rotation on consecutive failures."""
        initial_key = api_manager.get_current_key()
        
        # Mark consecutive failures
        for _ in range(3):
            api_manager.mark_key_failure(initial_key)
        
        # Should auto-rotate
        rotated = api_manager.auto_rotate_on_failure(initial_key, max_failures=2)
        
        assert rotated
        assert api_manager.get_current_key() != initial_key
    
    def test_empty_api_keys(self):
        """Test APIKeyManager with empty key list."""
        with pytest.raises(APIKeyError):
            APIKeyManager([])
    
    def test_single_api_key(self):
        """Test APIKeyManager with single key."""
        manager = APIKeyManager(["single_key"])
        
        key1 = manager.get_current_key()
        manager.rotate_key()
        key2 = manager.get_current_key()
        
        # Should return same key since there's only one
        assert key1 == key2 == "single_key"
