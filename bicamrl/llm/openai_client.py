"""OpenAI API client implementation."""

import os
from typing import Any, Dict, List, Optional

import aiohttp

from ..exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from ..utils.logging_config import get_logger
from .base_client import BaseLLMClient

logger = get_logger("llm.openai_client")


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (also supports OpenAI-compatible APIs like LM Studio)."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI client."""
        # Get API key from config or environment
        api_key = api_key or (config.get("api_key") if config else None)
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        # For LM Studio and other compatible APIs, api_key might not be needed
        if not api_key and config and config.get("base_url", "").startswith("http://localhost"):
            api_key = "not-needed"

        super().__init__(api_key, config)

        # Configuration
        self.base_url = self.config.get(
            "base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = self.config.get("model", "gpt-4-turbo-preview")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)

        logger.info(
            f"Initialized OpenAI client with model: {self.model}, base_url: {self.base_url}"
        )

    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make request to OpenAI API."""
        await self._ensure_session()

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # Build request data
        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add optional parameters
        if "response_format" in kwargs:
            data["response_format"] = kwargs["response_format"]

        if "stream" in kwargs:
            data["stream"] = kwargs["stream"]

        try:
            async with self._session.post(
                f"{self.base_url}/chat/completions", headers=headers, json=data
            ) as response:
                # Handle rate limiting
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    raise LLMRateLimitError(
                        "OpenAI rate limit exceeded",
                        retry_after=float(retry_after) if retry_after else None,
                    )

                # Handle other errors
                if response.status >= 400:
                    error_data = await response.text()
                    raise LLMResponseError(
                        f"OpenAI API error (status {response.status})",
                        provider="openai",
                        raw_response=error_data,
                    )

                # Parse successful response
                result = await response.json()

                # Extract content from response
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
                else:
                    raise LLMResponseError(
                        "Invalid response format from OpenAI",
                        provider="openai",
                        raw_response=str(result),
                    )

        except aiohttp.ClientError as e:
            raise LLMConnectionError(
                f"Failed to connect to OpenAI API: {str(e)}", {"base_url": self.base_url}
            )

    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze with JSON response format when possible."""
        # Request JSON format for structured analysis
        kwargs["response_format"] = {"type": "json_object"}
        return await super().analyze(prompt, **kwargs)

    async def stream_generate(self, prompt: str, callback: Optional[Any] = None, **kwargs) -> str:
        """Generate content with streaming support."""
        messages = self._build_generation_messages(prompt, **kwargs)
        kwargs["stream"] = True

        # For now, just return regular completion
        # TODO: Implement proper streaming support
        kwargs.pop("stream")
        return await self.complete(messages, **kwargs)
