"""Logging configuration for Bicamrl."""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def setup_logging(log_dir: str = ".bicamrl/logs", level: str = "INFO"):
    """
    Set up logging configuration for Bicamrl.

    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler for all logs (detailed format)
    log_file = log_path / f"bicameral_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Error file handler
    error_file = log_path / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)

    # Set specific logger levels
    logging.getLogger("bicameral").setLevel(logging.DEBUG)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Log startup
    logger = logging.getLogger("bicameral")
    logger.info(f"Logging initialized - Level: {level}, Directory: {log_path}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"bicamrl.{name}")


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs for production debugging."""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.thread,
            "thread_name": record.threadName,
        }
        
        # Add extra fields if present
        if hasattr(record, 'interaction_id'):
            log_entry['interaction_id'] = record.interaction_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'user_query'):
            log_entry['user_query'] = record.user_query
        if hasattr(record, 'action'):
            log_entry['action'] = record.action
        if hasattr(record, 'pattern_id'):
            log_entry['pattern_id'] = record.pattern_id
        if hasattr(record, 'memory_type'):
            log_entry['memory_type'] = record.memory_type
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        if hasattr(record, 'stack_trace'):
            log_entry['stack_trace'] = record.stack_trace
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, default=str)


def setup_production_logging(log_dir: str = ".bicamrl/logs"):
    """Set up production-grade logging with structured output."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    structured_formatter = StructuredFormatter()
    human_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(human_formatter)
    root_logger.addHandler(console_handler)
    
    # Structured log file (JSON format for debugging)
    structured_file = log_path / f"bicamrl_{datetime.now().strftime('%Y%m%d')}_structured.jsonl"
    structured_handler = logging.handlers.RotatingFileHandler(
        structured_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
    )
    structured_handler.setLevel(logging.DEBUG)
    structured_handler.setFormatter(structured_formatter)
    root_logger.addHandler(structured_handler)
    
    # Human-readable debug file
    debug_file = log_path / f"bicamrl_{datetime.now().strftime('%Y%m%d')}_debug.log"
    debug_handler = logging.handlers.RotatingFileHandler(
        debug_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=5,
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)s - %(funcName)s:%(lineno)d - %(message)s\n"
        "  Thread: %(threadName)s | Process: %(processName)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root_logger.addHandler(debug_handler)
    
    # Performance log (for timing analysis)
    perf_file = log_path / "performance.jsonl"
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(structured_formatter)
    perf_handler.addFilter(lambda record: hasattr(record, 'duration_ms'))
    root_logger.addHandler(perf_handler)
    
    # Set specific logger levels
    logging.getLogger("bicamrl").setLevel(logging.DEBUG)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    return get_logger("main")
