"""
Rate limiter for API calls to respect API limits.

This module provides a token bucket-based rate limiter to ensure
API calls stay within allowed rate limits.
"""

import time
import threading
import logging
from typing import Dict, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket-based rate limiter for API calls.
    
    Implements a token bucket algorithm where tokens are added at a fixed rate
    and consumed by API calls. If no tokens are available, the call waits.
    
    Attributes:
        calls_per_second: Maximum number of calls allowed per second
        burst_size: Maximum burst size (number of tokens in bucket)
    """
    
    def __init__(
        self,
        calls_per_second: float = 1.0,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum number of calls allowed per second
            burst_size: Maximum burst size (defaults to calls_per_second)
        """
        self.calls_per_second = calls_per_second
        self.burst_size = burst_size or int(calls_per_second)
        
        # Token bucket state
        self._tokens = float(self.burst_size)
        self._last_update = time.time()
        self._lock = threading.Lock()
        
        logger.info(
            f"Initialized RateLimiter: {calls_per_second} calls/sec, "
            f"burst size: {self.burst_size}"
        )
    
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            Time waited in seconds
        """
        with self._lock:
            # Refill tokens based on time elapsed
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.calls_per_second
            )
            self._last_update = now
            
            # If not enough tokens, calculate wait time
            if self._tokens < tokens:
                wait_time = (tokens - self._tokens) / self.calls_per_second
                
                logger.debug(
                    f"Rate limit reached. Waiting {wait_time:.2f} seconds..."
                )
                
                time.sleep(wait_time)
                
                # Update state after waiting
                self._tokens = 0
                self._last_update = time.time()
                
                return wait_time
            
            # Consume tokens
            self._tokens -= tokens
            return 0.0
    
    def __call__(self, func):
        """
        Decorator to add rate limiting to a function.
        
        Args:
            func: Function to wrap with rate limiting
            
        Returns:
            Wrapped function with rate limiting
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        
        return wrapper
    
    def reset(self) -> None:
        """Reset rate limiter state (for testing purposes)."""
        with self._lock:
            self._tokens = float(self.burst_size)
            self._last_update = time.time()


class MultiRateLimiter:
    """
    Manages multiple rate limiters for different API sources.
    
    Allows setting different rate limits for different API endpoints
    or data sources.
    """
    
    def __init__(self):
        """Initialize multi-rate limiter."""
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
    
    def add_limiter(
        self,
        name: str,
        calls_per_second: float,
        burst_size: Optional[int] = None
    ) -> None:
        """
        Add a rate limiter for a specific source.
        
        Args:
            name: Identifier for the rate limiter (e.g., "reuters", "ice_london")
            calls_per_second: Maximum calls per second for this source
            burst_size: Maximum burst size (defaults to calls_per_second)
        """
        with self._lock:
            self._limiters[name] = RateLimiter(
                calls_per_second=calls_per_second,
                burst_size=burst_size
            )
            logger.info(f"Added rate limiter for '{name}'")
    
    def acquire(self, name: str, tokens: int = 1) -> float:
        """
        Acquire tokens from a specific rate limiter.
        
        Args:
            name: Identifier for the rate limiter
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited in seconds
            
        Raises:
            KeyError: If rate limiter with given name doesn't exist
        """
        if name not in self._limiters:
            raise KeyError(f"Rate limiter '{name}' not found")
        
        return self._limiters[name].acquire(tokens)
    
    def get_limiter(self, name: str) -> RateLimiter:
        """
        Get a specific rate limiter by name.
        
        Args:
            name: Identifier for the rate limiter
            
        Returns:
            RateLimiter instance
            
        Raises:
            KeyError: If rate limiter with given name doesn't exist
        """
        if name not in self._limiters:
            raise KeyError(f"Rate limiter '{name}' not found")
        
        return self._limiters[name]
    
    def reset_all(self) -> None:
        """Reset all rate limiters."""
        with self._lock:
            for limiter in self._limiters.values():
                limiter.reset()


def with_rate_limit(
    calls_per_second: float = 1.0,
    burst_size: Optional[int] = None
):
    """
    Convenience decorator for adding rate limiting to functions.
    
    Args:
        calls_per_second: Maximum calls per second
        burst_size: Maximum burst size
        
    Returns:
        Decorator function
        
    Example:
        @with_rate_limit(calls_per_second=2.0)
        def fetch_data():
            # API call
            pass
    """
    limiter = RateLimiter(
        calls_per_second=calls_per_second,
        burst_size=burst_size
    )
    return limiter
