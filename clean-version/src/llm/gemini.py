"""
Google Gemini LLM integration with API key rotation and error handling.
"""
import logging
import asyncio
import os
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .api_manager import APIKeyManager
from ..utils.exceptions import LLMError, APIError
from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiLLM:
    """Google Gemini LLM wrapper with API key rotation."""
    
    def __init__(self, api_keys: Optional[List[str]] = None, model_name: str = "gemini-1.5-flash"):
        # Get API keys from environment if not provided
        if api_keys is None:
            # Try to get from GOOGLE_API_KEYS (comma-separated list)
            api_keys_str = os.getenv("GOOGLE_API_KEYS", "")
            if api_keys_str:
                self.api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
            else:
                # Fall back to single key or settings
                single_key = os.getenv("GOOGLE_API_KEY", "")
                if single_key:
                    self.api_keys = [single_key]
                else:
                    self.api_keys = settings.valid_api_keys
        else:
            self.api_keys = api_keys
            
        if not self.api_keys:
            raise LLMError("No valid API keys provided for Gemini")
        
        self.model_name = model_name
        self.api_manager = APIKeyManager(self.api_keys)
        
        # Generation config
        self.generation_config = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "max_output_tokens": settings.max_tokens,
        }
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        logger.info(f"Initialized Gemini LLM with model: {model_name}")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using Gemini API with key rotation."""
        
        async def _generate_with_key(api_key: str, prompt: str, **kwargs) -> str:
            """Internal method to generate response with a specific API key."""
            try:
                # Configure API key
                genai.configure(api_key=api_key)
                
                # Create model instance
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                # Generate response
                response = await model.generate_content_async(
                    prompt,
                    **kwargs
                )
                
                if not response.text:
                    raise LLMError("Empty response from Gemini API")
                
                return response.text
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for specific error types
                if any(phrase in error_msg for phrase in ['rate limit', 'quota', 'too many requests']):
                    raise APIError(f"Rate limit exceeded: {str(e)}")
                elif 'api key' in error_msg or 'authentication' in error_msg:
                    raise APIError(f"Authentication error: {str(e)}")
                elif 'safety' in error_msg or 'blocked' in error_msg:
                    raise LLMError(f"Content blocked by safety filters: {str(e)}")
                else:
                    raise LLMError(f"Gemini API error: {str(e)}")
        
        try:
            return await self.api_manager.with_retry(_generate_with_key, prompt, **kwargs)
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")
    
    async def generate_batch_responses(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate multiple responses in batch."""
        tasks = [self.generate_response(prompt, **kwargs) for prompt in prompts]
        try:
            return await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Failed to generate batch responses: {str(e)}")
            raise LLMError(f"Failed to generate batch responses: {str(e)}")
    
    async def generate_with_context(
        self, 
        query: str, 
        context: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response with context and optional system prompt."""
        
        # Build the full prompt
        full_prompt = ""
        
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        
        full_prompt += f"Context: {context}\n\n"
        full_prompt += f"Question: {query}\n\n"
        full_prompt += "Please provide a comprehensive answer based on the context provided."
        
        return await self.generate_response(full_prompt, **kwargs)
    
    async def extract_information(self, text: str, extraction_prompt: str, **kwargs) -> str:
        """Extract specific information from text using a custom prompt."""
        
        full_prompt = f"{extraction_prompt}\n\nText to analyze:\n{text}"
        
        return await self.generate_response(full_prompt, **kwargs)
    
    async def summarize_text(self, text: str, max_length: int = 500, **kwargs) -> str:
        """Summarize text to a specific length."""
        
        prompt = f"""Please provide a concise summary of the following text in approximately {max_length} words:

Text: {text}

Summary:"""
        
        return await self.generate_response(prompt, **kwargs)
    
    async def answer_question(
        self, 
        question: str, 
        context: Optional[str] = None, 
        previous_conversation: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Answer a question with optional context and conversation history."""
        
        full_prompt = ""
        
        # Add conversation history if provided
        if previous_conversation:
            full_prompt += "Previous conversation:\n"
            for turn in previous_conversation[-5:]:  # Last 5 turns
                full_prompt += f"Human: {turn.get('human', '')}\n"
                full_prompt += f"Assistant: {turn.get('assistant', '')}\n\n"
        
        # Add context if provided
        if context:
            full_prompt += f"Context: {context}\n\n"
        
        # Add the current question
        full_prompt += f"Question: {question}\n\n"
        full_prompt += "Please provide a helpful and accurate answer."
        
        return await self.generate_response(full_prompt, **kwargs)
    
    async def evaluate_relevance(self, query: str, text: str, **kwargs) -> Dict[str, Any]:
        """Evaluate the relevance of text to a query."""
        
        prompt = f"""Evaluate how relevant the following text is to the given query.

Query: {query}

Text: {text}

Please provide:
1. A relevance score from 0-10 (where 10 is most relevant)
2. A brief explanation of the relevance
3. Key matching concepts

Format your response as:
Score: [0-10]
Explanation: [brief explanation]
Key concepts: [list of matching concepts]"""
        
        response = await self.generate_response(prompt, **kwargs)
        
        # Parse the response (simplified parsing)
        lines = response.split('\n')
        result = {'score': 0, 'explanation': '', 'key_concepts': []}
        
        for line in lines:
            if line.startswith('Score:'):
                try:
                    score_str = line.split(':')[1].strip()
                    result['score'] = float(score_str)
                except:
                    pass
            elif line.startswith('Explanation:'):
                result['explanation'] = line.split(':', 1)[1].strip()
            elif line.startswith('Key concepts:'):
                concepts_str = line.split(':', 1)[1].strip()
                result['key_concepts'] = [c.strip() for c in concepts_str.split(',')]
        
        return result
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return self.api_manager.get_key_stats()
    
    async def test_connection(self) -> bool:
        """Test if the API connection is working."""
        try:
            test_response = await self.generate_response("Hello, please respond with 'API connection successful'.")
            return "successful" in test_response.lower()
        except Exception as e:
            logger.error(f"API connection test failed: {str(e)}")
            return False
