"""
Retry handler with exponential backoff for API calls.

This module provides a robust retry mechanism for handling transient failures
when collecting data from external APIs.
"""

import time
import logging
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class RetryHandler:
    """
    Handles retries with exponential backoff for API calls.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff (default: 2)
        retry_delay: Initial delay in seconds before first retry
        retryable_exceptions: Tuple of exception types that should trigger retry
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        retry_delay: float = 5.0,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            backoff_factor: Multiplier for exponential backoff (default: 2.0)
            retry_delay: Initial delay in seconds before first retry (default: 5.0)
            retryable_exceptions: Tuple of exception types to retry on
                                 (default: Exception for all exceptions)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_delay = retry_delay
        self.retryable_exceptions = retryable_exceptions or (Exception,)
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to add retry logic to a function.
        
        Args:
            func: Function to wrap with retry logic
            
        Returns:
            Wrapped function with retry capability
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return self.execute_with_retry(func, *args, **kwargs)
        
        return wrapper
    
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: The last exception if all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # Log success if this was a retry
                if attempt > 0:
                    logger.info(
                        f"Successfully executed {func.__name__} on attempt {attempt + 1}"
                    )
                
                return result
                
            except self.retryable_exceptions as e:
                last_exception = e
                
                # Don't retry if we've exhausted attempts
                if attempt >= self.max_retries:
                    logger.error(
                        f"Failed to execute {func.__name__} after {self.max_retries + 1} attempts: {str(e)}"
                    )
                    break
                
                # Calculate delay with exponential backoff
                delay = self.retry_delay * (self.backoff_factor ** attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                
                time.sleep(delay)
        
        # Raise the last exception if all retries failed
        raise last_exception
    
    def reset(self) -> None:
        """Reset retry handler state (for testing purposes)."""
        pass  # Stateless implementation, nothing to reset


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    retry_delay: float = 5.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
) -> Callable:
    """
    Convenience decorator for adding retry logic to functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        retry_delay: Initial delay in seconds before first retry
        retryable_exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorator function
        
    Example:
        @with_retry(max_retries=3, retry_delay=5.0)
        def fetch_data():
            # API call that might fail
            pass
    """
    handler = RetryHandler(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        retry_delay=retry_delay,
        retryable_exceptions=retryable_exceptions
    )
    return handler
