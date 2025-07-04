"""Custom exceptions for Bicamrl system."""

from typing import Any, Dict, Optional


class BicamrlError(Exception):
    """Base exception for all Bicamrl errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class MemoryError(BicamrlError):
    """Error related to memory operations."""

    pass


class StorageError(BicamrlError):
    """Error related to storage operations."""

    pass


class DatabaseError(StorageError):
    """Error related to database operations."""

    pass


class JSONParsingError(StorageError):
    """Error parsing JSON data."""

    def __init__(
        self, message: str, data: Optional[str] = None, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.data = data


class LLMError(BicamrlError):
    """Error related to LLM operations."""

    pass


class LLMConnectionError(LLMError):
    """Error connecting to LLM service."""

    pass


class LLMResponseError(LLMError):
    """Error in LLM response."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        raw_response: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.provider = provider
        self.raw_response = raw_response


class LLMRateLimitError(LLMError):
    """Rate limit exceeded for LLM service."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


class ConfigurationError(BicamrlError):
    """Error in configuration."""

    pass


class PatternDetectionError(BicamrlError):
    """Error in pattern detection."""

    pass


class ConsolidationError(BicamrlError):
    """Error during memory consolidation."""

    pass


class WorldModelError(BicamrlError):
    """Error in world model operations."""

    pass


class SleepLayerError(BicamrlError):
    """Error in sleep layer operations."""

    pass


class ImportError(BicamrlError):
    """Error importing data."""

    def __init__(
        self,
        message: str,
        source_file: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.source_file = source_file


class ExportError(BicamrlError):
    """Error exporting data."""

    def __init__(
        self,
        message: str,
        target_file: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.target_file = target_file
