"""MCP-specific logging decorators for tools and resources."""

import time
import functools
import traceback
from typing import Any, Dict, Optional, Callable
from datetime import datetime

from .logging_config import get_logger
from .log_utils import create_interaction_logger as create_log_context


def mcp_tool_logger(tool_name: Optional[str] = None):
    """
    Decorator for MCP tools that provides comprehensive logging.
    
    Features:
    - Logs tool invocation with all parameters
    - Tracks execution time
    - Logs results and errors
    - Creates interaction-specific logging context when applicable
    - Provides structured logging for production debugging
    
    Args:
        tool_name: Optional override for tool name (defaults to function name)
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get tool name and logger
            name = tool_name or func.__name__
            logger = get_logger(f"mcp.tool.{name}")
            start_time = time.time()
            
            # Extract key parameters for logging
            extra = {
                'tool': name,
                'timestamp': datetime.now().isoformat(),
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            # Add specific parameter details based on tool
            if name == "start_interaction":
                extra['query_length'] = len(kwargs.get('user_query', ''))
                extra['has_context'] = kwargs.get('context') is not None
            elif name == "log_ai_interpretation":
                extra['interaction_id'] = kwargs.get('interaction_id')
                extra['confidence'] = kwargs.get('confidence')
                extra['active_role'] = kwargs.get('active_role')
            elif name == "log_action":
                extra['action_type'] = kwargs.get('action_type')
                extra['target'] = kwargs.get('target')
            elif name == "complete_interaction":
                extra['has_feedback'] = kwargs.get('feedback') is not None
                extra['success'] = kwargs.get('success')
            elif name == "detect_pattern":
                extra['sequence_length'] = len(args[0]) if args else 0
            elif name == "get_relevant_context":
                extra['task_preview'] = kwargs.get('task_description', '')[:100]
                extra['file_context_count'] = len(kwargs.get('file_context', []))
            elif name == "record_feedback":
                extra['feedback_type'] = kwargs.get('feedback_type')
                extra['message_length'] = len(kwargs.get('message', ''))
                extra['interaction_id'] = kwargs.get('interaction_id')
            
            # Add session info if available
            try:
                from ..server import memory
                if memory:
                    extra['session_id'] = memory.session_id
            except:
                pass
            
            logger.info(f"Tool '{name}' invoked", extra=extra)
            
            try:
                # Execute the tool
                result = await func(*args, **kwargs)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                extra['duration_ms'] = duration_ms
                
                # Log result summary
                if isinstance(result, dict):
                    extra['result_keys'] = list(result.keys())
                    if 'error' in result:
                        extra['has_error'] = True
                        logger.warning(
                            f"Tool '{name}' completed with error",
                            extra={**extra, 'error': result.get('error')}
                        )
                    else:
                        extra['has_error'] = False
                        logger.info(f"Tool '{name}' completed successfully", extra=extra)
                elif isinstance(result, str):
                    extra['result_length'] = len(result)
                    extra['result_preview'] = result[:100]
                    logger.info(f"Tool '{name}' completed", extra=extra)
                else:
                    extra['result_type'] = type(result).__name__
                    logger.info(f"Tool '{name}' completed", extra=extra)
                
                return result
                
            except Exception as e:
                # Calculate duration even on error
                duration_ms = (time.time() - start_time) * 1000
                extra['duration_ms'] = duration_ms
                extra['error_type'] = type(e).__name__
                extra['error_message'] = str(e)
                extra['stack_trace'] = traceback.format_exc()
                
                logger.error(
                    f"Tool '{name}' failed with {type(e).__name__}",
                    extra=extra,
                    exc_info=True
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator


def mcp_resource_logger(resource_name: Optional[str] = None):
    """
    Decorator for MCP resources that provides logging.
    
    Features:
    - Logs resource access
    - Tracks retrieval time
    - Logs data size and type
    - Handles errors gracefully
    
    Args:
        resource_name: Optional override for resource name
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get resource name and logger
            name = resource_name or func.__name__
            logger = get_logger(f"mcp.resource.{name}")
            start_time = time.time()
            
            extra = {
                'resource': name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add session info if available
            try:
                from ..server import memory
                if memory:
                    extra['session_id'] = memory.session_id
            except:
                pass
            
            logger.debug(f"Resource '{name}' accessed", extra=extra)
            
            try:
                # Get the resource
                result = await func(*args, **kwargs)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                extra['duration_ms'] = duration_ms
                
                # Log resource details
                if hasattr(result, 'text'):
                    extra['content_length'] = len(result.text)
                    extra['mime_type'] = getattr(result, 'mimeType', 'unknown')
                    
                logger.info(f"Resource '{name}' retrieved", extra=extra)
                
                return result
                
            except Exception as e:
                # Calculate duration even on error
                duration_ms = (time.time() - start_time) * 1000
                extra['duration_ms'] = duration_ms
                extra['error_type'] = type(e).__name__
                extra['error_message'] = str(e)
                
                logger.error(
                    f"Resource '{name}' failed",
                    extra=extra,
                    exc_info=True
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator


def with_interaction_logging(func):
    """
    Decorator that creates an interaction-specific logger for tools that work with interactions.
    
    This decorator:
    - Extracts interaction_id from function arguments
    - Creates an interaction-specific logger
    - Makes it available as 'interaction_log' in the function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Try to extract interaction_id
        interaction_id = kwargs.get('interaction_id')
        
        # If not in kwargs, check if it's in current interaction
        if not interaction_id:
            try:
                from ..server import interaction_logger
                if interaction_logger and interaction_logger.current_interaction:
                    interaction_id = interaction_logger.current_interaction.interaction_id
            except:
                pass
        
        # Create interaction logger if we have an ID
        if interaction_id:
            try:
                from ..server import memory
                session_id = memory.session_id if memory else None
                interaction_log = create_log_context(interaction_id, session_id)
                
                # Add to kwargs so function can use it
                kwargs['_interaction_log'] = interaction_log
            except:
                pass
        
        return await func(*args, **kwargs)
    
    return wrapper


def log_tool_metric(metric_name: str, value: Any, tool_name: str, **extra):
    """
    Helper function to log tool-specific metrics.
    
    Args:
        metric_name: Name of the metric (e.g., "patterns_found", "similarity_score")
        value: The metric value
        tool_name: Name of the tool
        **extra: Additional context to log
    """
    logger = get_logger(f"mcp.tool.{tool_name}.metrics")
    
    log_entry = {
        'metric': metric_name,
        'value': value,
        'tool': tool_name,
        'timestamp': datetime.now().isoformat(),
        **extra
    }
    
    logger.info(f"Tool metric: {metric_name}={value}", extra=log_entry)