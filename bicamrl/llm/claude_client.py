"""Claude API client implementation."""

import os
from typing import Any, Dict, List, Optional

import aiohttp

from ..exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from ..utils.logging_config import get_logger
from .base_client import BaseLLMClient

logger = get_logger("llm.claude_client")


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude API client."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize Claude client."""
        # Get API key from config or environment
        api_key = api_key or (config.get("api_key") if config else None)
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("Claude API key required (set ANTHROPIC_API_KEY or provide api_key)")

        super().__init__(api_key, config)

        # Configuration
        self.base_url = self.config.get("base_url", "https://api.anthropic.com")
        self.model = self.config.get("model", "claude-3-opus-20240229")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)
        self.anthropic_version = self.config.get("anthropic_version", "2023-06-01")

        logger.info(f"Initialized Claude client with model: {self.model}")

    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make request to Claude API."""
        await self._ensure_session()

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }

        # Convert messages to Claude format
        system = None
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})

        # Build request data
        data = {
            "model": kwargs.get("model", self.model),
            "messages": claude_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if system:
            data["system"] = system

        try:
            async with self._session.post(
                f"{self.base_url}/v1/messages", headers=headers, json=data
            ) as response:
                # Handle rate limiting
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    raise LLMRateLimitError(
                        "Claude rate limit exceeded",
                        retry_after=float(retry_after) if retry_after else None,
                    )

                # Handle other errors
                if response.status >= 400:
                    error_data = await response.text()
                    raise LLMResponseError(
                        f"Claude API error (status {response.status})",
                        provider="claude",
                        raw_response=error_data,
                    )

                # Parse successful response
                result = await response.json()

                # Extract content from response
                if "content" in result and result["content"]:
                    # Claude returns content as array of blocks
                    content_blocks = result["content"]
                    if isinstance(content_blocks, list) and content_blocks:
                        # Concatenate text blocks
                        text_parts = []
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                text_parts.append(block)
                        return "".join(text_parts)

                raise LLMResponseError(
                    "Invalid response format from Claude",
                    provider="claude",
                    raw_response=str(result),
                )

        except aiohttp.ClientError as e:
            raise LLMConnectionError(
                f"Failed to connect to Claude API: {str(e)}", {"base_url": self.base_url}
            )
