"""
API key management for rotating between multiple API keys.
"""
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import time
from dataclasses import dataclass

from ..utils.exceptions import APIError

logger = logging.getLogger(__name__)


@dataclass
class APIKeyInfo:
    """Information about an API key."""
    key: str
    last_used: datetime
    usage_count: int
    is_rate_limited: bool
    rate_limit_reset: Optional[datetime]
    error_count: int


class APIKeyManager:
    """Manages rotation and rate limiting for API keys."""
    
    def __init__(self, api_keys: List[str], rate_limit_window: int = None):
        if not api_keys:
            raise APIError("No API keys provided")
        
        # Get rate limiting settings from environment
        import os
        self.rate_limit_per_key = int(os.getenv("RATE_LIMIT_PER_KEY", "100"))
        self.cooldown_after_limit = int(os.getenv("COOLDOWN_AFTER_LIMIT", "60"))
        self.min_request_interval = float(os.getenv("MIN_REQUEST_INTERVAL", "0.6"))
        
        # Use environment variable or fallback to parameter
        rate_limit_window = rate_limit_window or int(os.getenv("COOLDOWN_AFTER_LIMIT", "60"))
        
        self.api_keys = [
            APIKeyInfo(
                key=key,
                last_used=datetime.min,
                usage_count=0,
                is_rate_limited=False,
                rate_limit_reset=None,
                error_count=0
            )
            for key in api_keys if key and key.strip()
        ]
        
        if not self.api_keys:
            raise APIError("No valid API keys provided")
        
        self.rate_limit_window = rate_limit_window
        self.current_index = 0
        self.max_retries = 3
        self.retry_delay = 1.0
        
        logger.info(f"Initialized API key manager with {len(self.api_keys)} keys")
    
    def get_next_key(self) -> str:
        """Get the next available API key."""
        current_time = datetime.now()
        
        # First, try to find a key that's not rate limited
        for _ in range(len(self.api_keys)):
            key_info = self.api_keys[self.current_index]
            
            # Check if rate limit has expired
            if key_info.is_rate_limited and key_info.rate_limit_reset:
                if current_time >= key_info.rate_limit_reset:
                    key_info.is_rate_limited = False
                    key_info.rate_limit_reset = None
                    logger.info(f"Rate limit expired for key ending in ...{key_info.key[-4:]}")
            
            # If key is available, use it
            if not key_info.is_rate_limited:
                key_info.last_used = current_time
                key_info.usage_count += 1
                
                logger.debug(f"Using API key ending in ...{key_info.key[-4:]} (usage: {key_info.usage_count})")
                return key_info.key
            
            # Move to next key
            self.current_index = (self.current_index + 1) % len(self.api_keys)
        
        # All keys are rate limited, wait for the earliest reset
        earliest_reset = min(
            key.rate_limit_reset for key in self.api_keys 
            if key.rate_limit_reset is not None
        )
        
        if earliest_reset:
            wait_time = (earliest_reset - current_time).total_seconds()
            if wait_time > 0:
                logger.warning(f"All API keys rate limited. Waiting {wait_time:.1f} seconds")
                time.sleep(min(wait_time, 60))  # Don't wait more than 60 seconds
                return self.get_next_key()
        
        # If we get here, something is wrong
        raise APIError("No available API keys")
    
    def mark_key_rate_limited(self, api_key: str, reset_time: Optional[datetime] = None) -> None:
        """Mark an API key as rate limited."""
        for key_info in self.api_keys:
            if key_info.key == api_key:
                key_info.is_rate_limited = True
                key_info.rate_limit_reset = reset_time or (datetime.now() + timedelta(minutes=1))
                logger.warning(f"Marked key ending in ...{api_key[-4:]} as rate limited")
                break
    
    def mark_key_error(self, api_key: str) -> None:
        """Mark an API key as having an error."""
        for key_info in self.api_keys:
            if key_info.key == api_key:
                key_info.error_count += 1
                logger.warning(f"Error count for key ending in ...{api_key[-4:]}: {key_info.error_count}")
                
                # If too many errors, temporarily rate limit
                if key_info.error_count >= 3:
                    self.mark_key_rate_limited(api_key, datetime.now() + timedelta(minutes=5))
                break
    
    def reset_key_errors(self, api_key: str) -> None:
        """Reset error count for an API key."""
        for key_info in self.api_keys:
            if key_info.key == api_key:
                key_info.error_count = 0
                break
    
    def get_key_stats(self) -> Dict[str, Any]:
        """Get statistics about API key usage."""
        stats = {
            'total_keys': len(self.api_keys),
            'available_keys': sum(1 for key in self.api_keys if not key.is_rate_limited),
            'rate_limited_keys': sum(1 for key in self.api_keys if key.is_rate_limited),
            'total_usage': sum(key.usage_count for key in self.api_keys),
            'keys': []
        }
        
        for i, key_info in enumerate(self.api_keys):
            stats['keys'].append({
                'index': i,
                'key_suffix': key_info.key[-4:],
                'usage_count': key_info.usage_count,
                'error_count': key_info.error_count,
                'is_rate_limited': key_info.is_rate_limited,
                'last_used': key_info.last_used.isoformat() if key_info.last_used != datetime.min else None,
                'rate_limit_reset': key_info.rate_limit_reset.isoformat() if key_info.rate_limit_reset else None
            })
        
        return stats
    
    async def with_retry(self, operation, *args, **kwargs):
        """Execute an operation with retry logic using different API keys."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                api_key = self.get_next_key()
                result = await operation(api_key, *args, **kwargs)
                
                # Reset error count on success
                self.reset_key_errors(api_key)
                return result
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if any(phrase in error_msg for phrase in ['rate limit', 'quota', 'too many requests']):
                    self.mark_key_rate_limited(api_key)
                    logger.warning(f"Rate limit hit for key ending in ...{api_key[-4:]}")
                else:
                    self.mark_key_error(api_key)
                    logger.error(f"Error with key ending in ...{api_key[-4:]}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
        
        raise last_exception or APIError("All retry attempts failed")
