"""Base LLM client class with common functionality."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from ..exceptions import LLMConnectionError, LLMError, LLMRateLimitError, LLMResponseError
from ..utils.logging_config import get_logger

logger = get_logger("llm.base_client")


class BaseLLMClient(ABC):
    """
    Base class for all LLM clients with common functionality.

    Provides:
    - Retry logic with exponential backoff
    - Rate limiting
    - Error handling
    - Response validation
    - Connection pooling
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """Initialize base LLM client."""
        self.api_key = api_key
        self.config = config or {}
        self.max_retries = max_retries
        self.timeout = timeout

        # Session management
        self._session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self._rate_limiter = None
        if "rate_limit" in self.config:
            from ..rate_limiter import RateLimiter

            self._rate_limiter = RateLimiter(**self.config["rate_limit"])

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self):
        """Ensure HTTP session is available."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic and exponential backoff.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            LLMError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting if configured
                if self._rate_limiter:
                    await self._rate_limiter.acquire()

                # Execute the function
                result = await func(*args, **kwargs)
                return result

            except aiohttp.ClientError as e:
                # Network errors
                last_error = LLMConnectionError(
                    f"Network error: {str(e)}",
                    {"attempt": attempt + 1, "error_type": type(e).__name__},
                )
                logger.warning(f"Network error on attempt {attempt + 1}/{self.max_retries}: {e}")

            except LLMRateLimitError as e:
                # Rate limit errors - use retry_after if provided
                last_error = e
                wait_time = e.retry_after or (2**attempt)
                logger.warning(
                    f"Rate limit on attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s"
                )
                await asyncio.sleep(wait_time)
                continue

            except LLMError as e:
                # Other LLM errors
                last_error = e
                logger.warning(f"LLM error on attempt {attempt + 1}/{self.max_retries}: {e}")

            except Exception as e:
                # Unexpected errors
                last_error = LLMError(
                    f"Unexpected error: {str(e)}",
                    {"attempt": attempt + 1, "error_type": type(e).__name__},
                )
                logger.error(
                    f"Unexpected error on attempt {attempt + 1}/{self.max_retries}", exc_info=True
                )

            # Calculate backoff time
            if attempt < self.max_retries - 1:
                wait_time = min(2**attempt, 60)  # Cap at 60 seconds
                logger.debug(f"Waiting {wait_time}s before retry")
                await asyncio.sleep(wait_time)

        # All retries failed
        logger.error(f"All {self.max_retries} attempts failed")
        raise last_error or LLMError("All retry attempts failed")

    @abstractmethod
    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Make the actual API request. Must be implemented by subclasses.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Response content as string

        Raises:
            LLMError: On request failure
        """
        pass

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Get completion from LLM with retry logic.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Completion text
        """
        return await self._retry_with_backoff(self._make_request, messages, **kwargs)

    async def analyze(self, prompt: str, **kwargs) -> str:
        """
        Analyze content using the LLM.

        Args:
            prompt: Analysis prompt
            **kwargs: Additional parameters

        Returns:
            Analysis result
        """
        messages = self._build_analysis_messages(prompt, **kwargs)
        return await self.complete(messages, **kwargs)

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate content using the LLM.

        Args:
            prompt: Generation prompt
            **kwargs: Additional parameters

        Returns:
            Generated content
        """
        messages = self._build_generation_messages(prompt, **kwargs)
        return await self.complete(messages, **kwargs)

    def _build_analysis_messages(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> List[Dict[str, str]]:
        """Build messages for analysis request."""
        system_prompt = (
            system or "You are an AI system analyzer. Provide detailed, structured analysis."
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    def _build_generation_messages(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> List[Dict[str, str]]:
        """Build messages for generation request."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Args:
            response: Raw response string

        Returns:
            Parsed JSON object

        Raises:
            LLMResponseError: If parsing fails
        """
        import json

        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from response
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json_str = response[start_idx : end_idx + 1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Failed to parse
        raise LLMResponseError(
            "Failed to parse JSON from response",
            raw_response=response[:200],  # First 200 chars for debugging
        )
