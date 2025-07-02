"""Logging utilities for enhanced debugging."""

import time
import functools
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Callable
from ..utils.logging_config import get_logger


class LogContext:
    """Context manager for structured logging with timing."""
    
    def __init__(self, logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.extra = kwargs
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(
            f"Starting {self.operation}",
            extra=self.extra
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.extra['duration_ms'] = duration_ms
        
        if exc_type is not None:
            self.extra['error_type'] = exc_type.__name__
            self.extra['stack_trace'] = traceback.format_exc()
            self.logger.error(
                f"Failed {self.operation} after {duration_ms:.2f}ms: {exc_val}",
                extra=self.extra,
                exc_info=True
            )
        else:
            self.logger.debug(
                f"Completed {self.operation} in {duration_ms:.2f}ms",
                extra=self.extra
            )
        return False


@asynccontextmanager
async def async_log_context(logger, operation: str, **kwargs):
    """Async context manager for structured logging with timing."""
    start_time = time.time()
    extra = kwargs
    
    logger.debug(f"Starting {operation}", extra=extra)
    
    try:
        yield
        duration_ms = (time.time() - start_time) * 1000
        extra['duration_ms'] = duration_ms
        logger.debug(
            f"Completed {operation} in {duration_ms:.2f}ms",
            extra=extra
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        extra['duration_ms'] = duration_ms
        extra['error_type'] = type(e).__name__
        extra['stack_trace'] = traceback.format_exc()
        logger.error(
            f"Failed {operation} after {duration_ms:.2f}ms: {e}",
            extra=extra,
            exc_info=True
        )
        raise


def log_method_call(logger=None):
    """Decorator to log method calls with arguments and timing."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = getattr(self, 'logger', get_logger(self.__class__.__name__))
                
            func_name = f"{self.__class__.__name__}.{func.__name__}"
            extra = {
                'method': func_name,
                'args': str(args)[:200],  # Truncate long args
                'kwargs': str(kwargs)[:200]
            }
            
            async with async_log_context(logger, f"method {func_name}", **extra):
                result = await func(self, *args, **kwargs)
                return result
                
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = getattr(self, 'logger', get_logger(self.__class__.__name__))
                
            func_name = f"{self.__class__.__name__}.{func.__name__}"
            extra = {
                'method': func_name,
                'args': str(args)[:200],
                'kwargs': str(kwargs)[:200]
            }
            
            with LogContext(logger, f"method {func_name}", **extra):
                result = func(self, *args, **kwargs)
                return result
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def log_memory_operation(memory_type: str, operation: str):
    """Decorator specifically for memory operations."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            logger = getattr(self, 'logger', get_logger(self.__class__.__name__))
            
            extra = {
                'memory_type': memory_type,
                'memory_operation': operation,
                'method': func.__name__
            }
            
            # Add interaction_id if present in args
            if args and hasattr(args[0], 'interaction_id'):
                extra['interaction_id'] = args[0].interaction_id
            elif 'interaction_id' in kwargs:
                extra['interaction_id'] = kwargs['interaction_id']
                
            async with async_log_context(logger, f"memory {operation}", **extra):
                result = await func(self, *args, **kwargs)
                
                # Log result summary
                if result:
                    if isinstance(result, list):
                        logger.debug(f"Retrieved {len(result)} items", extra=extra)
                    elif isinstance(result, dict):
                        logger.debug(f"Result keys: {list(result.keys())}", extra=extra)
                        
                return result
                
        return wrapper
    return decorator


def log_pattern_operation(operation: str):
    """Decorator for pattern detection operations."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            logger = getattr(self, 'logger', get_logger(self.__class__.__name__))
            
            extra = {
                'pattern_operation': operation,
                'method': func.__name__
            }
            
            # Extract pattern info from args
            if args and hasattr(args[0], '__iter__'):
                extra['sequence_length'] = len(args[0])
                
            async with async_log_context(logger, f"pattern {operation}", **extra):
                result = await func(self, *args, **kwargs)
                
                # Log pattern matches
                if result and isinstance(result, list):
                    logger.info(
                        f"Found {len(result)} pattern matches",
                        extra={'matches': len(result), **extra}
                    )
                    for i, pattern in enumerate(result[:3]):  # Log first 3
                        if isinstance(pattern, dict):
                            logger.debug(
                                f"Pattern {i}: {pattern.get('name', 'unnamed')} "
                                f"(confidence: {pattern.get('confidence', 0):.2f})",
                                extra={'pattern_id': pattern.get('id'), **extra}
                            )
                            
                return result
                
        return wrapper
    return decorator


def create_interaction_logger(interaction_id: str, session_id: str):
    """Create a logger with interaction context pre-configured."""
    logger = get_logger("interaction")
    
    class InteractionLogger:
        def __init__(self):
            self.extra = {
                'interaction_id': interaction_id,
                'session_id': session_id
            }
            
        def debug(self, msg, **kwargs):
            logger.debug(msg, extra={**self.extra, **kwargs})
            
        def info(self, msg, **kwargs):
            logger.info(msg, extra={**self.extra, **kwargs})
            
        def warning(self, msg, **kwargs):
            logger.warning(msg, extra={**self.extra, **kwargs})
            
        def error(self, msg, **kwargs):
            logger.error(msg, extra={**self.extra, **kwargs})
            
        def log_action(self, action: str, target: str, details: Dict[str, Any]):
            self.info(
                f"Action: {action} on {target}",
                action=action,
                target=target,
                details=details
            )
            
        def log_query(self, query: str, interpretation: str = None):
            self.info(
                f"User query: {query[:100]}...",
                user_query=query,
                interpretation=interpretation
            )
            
        def log_feedback(self, feedback_type: str, content: str):
            self.info(
                f"Feedback ({feedback_type}): {content[:100]}...",
                feedback_type=feedback_type,
                feedback_content=content
            )
            
    return InteractionLogger()


import asyncio