"""LLM provider implementations for Sleep Layer."""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.api_key = api_key
        self.config = config or {}
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)

    @abstractmethod
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content and return insights."""
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content based on prompt."""
        pass

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = 2**attempt
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)


class ClaudeLLMProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"), config)
        self.base_url = "https://api.anthropic.com/v1"
        self.model = self.config.get("model", "claude-opus-4-20250514")
        self.max_tokens = self.config.get("max_tokens", 4096)

    async def _make_request(
        self, messages: List[Dict[str, str]], system: Optional[str] = None
    ) -> str:
        """Make a request to Claude API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        data = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}

        if system:
            data["system"] = system

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["content"][0]["text"]

    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content using Claude."""
        system = kwargs.get(
            "system", "You are an AI system analyzer. Provide detailed, structured analysis."
        )
        messages = [{"role": "user", "content": prompt}]

        return await self._retry_with_backoff(self._make_request, messages, system)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using Claude."""
        system = kwargs.get("system", "You are a helpful AI assistant.")
        messages = [{"role": "user", "content": prompt}]

        return await self._retry_with_backoff(self._make_request, messages, system)


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI GPT provider (also supports OpenAI-compatible APIs like LM Studio)."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"), config)
        # Support custom base URL for OpenAI-compatible APIs (e.g., LM Studio)
        self.base_url = self.config.get(
            "base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = self.config.get("model", "gpt-4-turbo-preview")
        self.max_tokens = self.config.get("max_tokens", 4096)

    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a request to OpenAI API."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
        }

        if "response_format" in kwargs:
            data["response_format"] = kwargs["response_format"]

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]

    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content using GPT."""
        messages = [
            {
                "role": "system",
                "content": "You are an AI system analyzer. Provide detailed, structured analysis.",
            },
            {"role": "user", "content": prompt},
        ]

        # Try to get JSON response for analysis
        kwargs["response_format"] = {"type": "json_object"}

        return await self._retry_with_backoff(self._make_request, messages, **kwargs)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using GPT."""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

        return await self._retry_with_backoff(self._make_request, messages, **kwargs)


class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider (e.g., Ollama, LlamaCpp)."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        self.base_url = self.config.get("base_url", "http://localhost:11434")
        self.model = self.config.get("model", "llama2")

    async def _make_request(self, prompt: str, **kwargs) -> str:
        """Make a request to local LLM."""
        data = {"model": self.model, "prompt": prompt, "stream": False}

        if "system" in kwargs:
            data["system"] = kwargs["system"]

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["response"]

    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content using local LLM."""
        system = "You are an AI system analyzer. Provide detailed, structured analysis."
        full_prompt = f"{system}\n\n{prompt}"

        return await self._retry_with_backoff(self._make_request, full_prompt, **kwargs)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using local LLM."""
        return await self._retry_with_backoff(self._make_request, prompt, **kwargs)


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing without external dependencies."""

    async def analyze(self, prompt: str, **kwargs) -> str:
        """Mock analysis with context-aware responses."""
        # Provide different responses based on prompt content
        if "world model" in prompt.lower() or "analyze this interaction" in prompt.lower():
            return json.dumps(
                {
                    "domain": "software_development",
                    "confidence": 0.8,
                    "entities": [
                        {"id": "file_1", "type": "source_file", "properties": {"path": "test.py"}}
                    ],
                    "relations": [{"source": "file_1", "target": "user", "type": "edits"}],
                    "goals": [
                        {"type": "bug_fix", "description": "Fix the issue", "confidence": 0.7}
                    ],
                }
            )
        elif "pattern" in prompt.lower():
            return json.dumps(
                {
                    "patterns": [
                        {
                            "type": "workflow",
                            "description": "Test pattern detected",
                            "frequency": 5,
                            "recommendation": "Consider automating this workflow",
                            "confidence": 0.8,
                        }
                    ]
                }
            )
        elif "consolidate" in prompt.lower() or "work session" in prompt.lower():
            return json.dumps(
                {
                    "sessions": [
                        {
                            "interactions": [0, 1, 2],
                            "theme": "debugging authentication",
                            "duration_minutes": 30,
                        }
                    ],
                    "insights": "User frequently debugs authentication issues",
                }
            )
        else:
            # Default analysis response
            return json.dumps({"analysis": "Mock analysis completed", "confidence": 0.75})

    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock generation with context-aware responses."""
        # Add a small delay to simulate processing
        await asyncio.sleep(2.0)  # 2 second delay to see thinking animation

        # Check for conversation context
        if "remember" in prompt.lower() and "User: Hello" in prompt:
            return "Yes, I remember! You said **'Hello'** to me just now."
        elif "ascii" in prompt.lower() or "draw" in prompt.lower():
            # Return ASCII art
            return """Here's a simple ASCII cat for you:

```
    /\\_/\\
   ( o.o )
    > ^ <
```

And here's a house:

```
      /\\
     /  \\
    /    \\
   /------\\
   |  []  |
   |______|
```

ASCII art looks best in monospace fonts!"""
        elif "hello" in prompt.lower() or "test" in prompt.lower():
            return """# Hello there! 👋

This is a **mock response** with *markdown formatting* to demonstrate the TUI's new capabilities:

## Features:
- **Bold text** for emphasis
- *Italic text* for subtle highlights
- `inline code` snippets
- Code blocks with syntax highlighting
- Unicode support: 你好, مرحبا, こんにちは, 🌍
- Emoji support: 😀 🚀 ❤️ ✨ 🎯 💻

```python
def greet(name):
    # This is a comment with emoji! 🐍
    return f"Hello, {name}! 👋"

result = greet("World 🌍")
print(result)  # Output: Hello, World 🌍! 👋
```

### Lists with Unicode
1. Numbered lists work too 📝
2. With multiple items
   - And nested bullets ▶️
   - Mathematical symbols: ∑ ∏ ∫ ≈ ≠ ≤ ≥
   - Arrows: ← → ↑ ↓ ⇐ ⇒ ⇑ ⇓

### Special Characters
- Box drawing: ┌─┬─┐ │ ├─┼─┤ └─┴─┘
- Stars: ★ ☆ ✦ ✧ ⭐
- Check marks: ✓ ✗ ☑ ☒
- Music: ♪ ♫ ♬ ♭ ♮ ♯

Enjoy the enhanced formatting! 🎨✨"""
        elif "enhance" in prompt.lower() or "improve" in prompt.lower():
            return "Enhanced: " + prompt[:100]
        elif "say 'ok'" in prompt.lower():
            return "OK"
        else:
            return "Mock response for: " + prompt[:50] + "..."


class ClaudeCodeProvider(BaseLLMProvider):
    """Claude Code SDK provider for using Claude via the official SDK."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        config = config or {}

        # Claude Code needs ANTHROPIC_API_KEY in environment
        import os

        if not os.getenv("ANTHROPIC_API_KEY") and not api_key:
            logger.warning("ANTHROPIC_API_KEY not set. Claude Code may not work properly.")

        # Get Claude CLI path from config
        self.claude_cli_path = config.get("claude_cli_path", "claude")

        # Get model from config, default to Claude 4 Opus
        self.model = config.get("model", "claude-opus-4-20250514")

        # Permission mode configuration
        logger.debug(f"ClaudeCodeProvider config: {config}")
        self.permission_mode = config.get("permission_mode", "default")

        # Validate permission_mode
        valid_modes = ["default", "acceptEdits", "bypassPermissions"]
        if self.permission_mode not in valid_modes:
            logger.error(
                f"Invalid permission_mode: {self.permission_mode}. Must be one of {valid_modes}"
            )
            logger.error(f"Falling back to 'default' mode")
            self.permission_mode = "default"
        self.permission_prompt_tool = config.get("permission_prompt_tool", None)
        self.mcp_servers = config.get("mcp_servers", {})
        self.mcp_tools = config.get("mcp_tools", [])
        self.allowed_tools = config.get("allowed_tools", [])  # Store allowed_tools

        # Store config for later use
        self.config = config

        # Conversation configuration
        self.max_turns = config.get("max_turns", None)  # Default to None (unlimited)

        # Only log in debug mode
        logger.debug(f"Claude Code provider initialized with model: {self.model}")
        logger.debug(f"Permission mode: {self.permission_mode}")
        logger.debug(f"Max turns: {self.max_turns if self.max_turns is not None else 'unlimited'}")
        logger.debug(f"MCP servers: {list(self.mcp_servers.keys())}")
        logger.debug(f"MCP tools: {self.mcp_tools}")
        logger.debug(f"Permission prompt tool: {self.permission_prompt_tool}")

        self._check_dependencies()

    def _check_dependencies(self):
        """Check if Claude Code SDK is installed."""
        try:
            import claude_code_sdk  # noqa: F401
        except ImportError:
            raise ImportError(
                "claude-code-sdk not installed. Install with: pip install claude-code-sdk"
            )

        # Check if Claude CLI is available
        import os
        import shutil

        # First try the configured path
        if os.path.isabs(self.claude_cli_path) and os.path.exists(self.claude_cli_path):
            # Absolute path provided and exists
            if not os.access(self.claude_cli_path, os.X_OK):
                raise RuntimeError(f"Claude CLI at {self.claude_cli_path} is not executable")
        elif shutil.which(self.claude_cli_path):
            # Relative path or command name found in PATH
            pass
        else:
            raise RuntimeError(
                f"Claude CLI not found at '{self.claude_cli_path}'. "
                "Set 'claude_cli_path' in the provider config or ensure 'claude' is in PATH"
            )

    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content using Claude Code."""
        system = kwargs.get(
            "system", "You are an AI system analyzer. Provide detailed, structured analysis."
        )

        try:
            # Use Claude Code SDK
            from claude_code_sdk import ClaudeCodeOptions, query

            # Get allowed_tools from config, default to empty list
            allowed_tools = self.config.get("allowed_tools", [])

            # Debug logging
            logger.info(f"Creating ClaudeCodeOptions with:")
            logger.info(
                f"  permission_mode='{self.permission_mode}' (type: {type(self.permission_mode)})"
            )
            logger.info(f"  permission_prompt_tool_name='{self.permission_prompt_tool}'")
            logger.info(f"  allowed_tools={allowed_tools}")
            logger.info(
                f"  mcp_servers={list(self.mcp_servers.keys()) if self.mcp_servers else []}"
            )
            logger.info(f"  mcp_tools={self.mcp_tools}")

            options = ClaudeCodeOptions(
                max_turns=self.max_turns,
                allowed_tools=allowed_tools,  # Use configured allowed_tools
                permission_mode=self.permission_mode,
                permission_prompt_tool_name=self.permission_prompt_tool,
                mcp_servers=self.mcp_servers,
                mcp_tools=self.mcp_tools,
                system_prompt=system,  # Use system prompt properly
                model=self.model,  # Use configured model
            )

            # Log the options object to debug
            logger.info(f"ClaudeCodeOptions object created:")
            logger.info(f"  options.permission_mode = {options.permission_mode}")
            logger.info(
                f"  options.permission_prompt_tool_name = {options.permission_prompt_tool_name}"
            )

            response_parts = []
            async for message in query(prompt=prompt, options=options):  # Use keyword arguments
                # Collect all response parts
                if hasattr(message, "content"):
                    # Handle content that might be a list of blocks
                    if isinstance(message.content, list):
                        for block in message.content:
                            if hasattr(block, "text"):
                                response_parts.append(block.text)
                            elif isinstance(block, str):
                                response_parts.append(block)
                    elif isinstance(message.content, str):
                        response_parts.append(message.content)

            return "".join(response_parts)
        except Exception as e:
            logger.error(f"Claude Code analyze failed: {e}")
            raise

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using Claude Code."""
        system = kwargs.get("system", "")

        try:
            from claude_code_sdk import ClaudeCodeOptions, query

            # Get allowed_tools from config, default to empty list
            allowed_tools = self.config.get("allowed_tools", [])

            # Debug logging
            logger.info(f"Creating ClaudeCodeOptions with:")
            logger.info(
                f"  permission_mode='{self.permission_mode}' (type: {type(self.permission_mode)})"
            )
            logger.info(f"  permission_prompt_tool_name='{self.permission_prompt_tool}'")
            logger.info(f"  allowed_tools={allowed_tools}")
            logger.info(
                f"  mcp_servers={list(self.mcp_servers.keys()) if self.mcp_servers else []}"
            )
            logger.info(f"  mcp_tools={self.mcp_tools}")

            options = ClaudeCodeOptions(
                max_turns=self.max_turns,
                allowed_tools=allowed_tools,  # Use configured allowed_tools
                permission_mode=self.permission_mode,
                permission_prompt_tool_name=self.permission_prompt_tool,
                mcp_servers=self.mcp_servers,
                mcp_tools=self.mcp_tools,
                system_prompt=system if system else None,
                model=self.model,  # Use configured model
            )

            response_parts = []
            async for message in query(prompt=prompt, options=options):  # Use keyword arguments
                if hasattr(message, "content"):
                    # Handle content that might be a list of blocks
                    if isinstance(message.content, list):
                        for block in message.content:
                            if hasattr(block, "text"):
                                response_parts.append(block.text)
                            elif isinstance(block, str):
                                response_parts.append(block)
                    elif isinstance(message.content, str):
                        response_parts.append(message.content)

            return "".join(response_parts)
        except Exception as e:
            logger.error(f"Claude Code generate failed: {e}")
            raise

    async def generate_stream(self, prompt: str, streaming_handler=None, **kwargs) -> str:
        """Generate content using Claude Code with streaming support."""
        system = kwargs.get("system", "")

        try:
            from claude_code_sdk import ClaudeCodeOptions, query

            # Get allowed_tools from config, default to empty list
            allowed_tools = self.config.get("allowed_tools", [])

            # Debug logging
            logger.info(f"Creating ClaudeCodeOptions with:")
            logger.info(
                f"  permission_mode='{self.permission_mode}' (type: {type(self.permission_mode)})"
            )
            logger.info(f"  permission_prompt_tool_name='{self.permission_prompt_tool}'")
            logger.info(f"  allowed_tools={allowed_tools}")
            logger.info(
                f"  mcp_servers={list(self.mcp_servers.keys()) if self.mcp_servers else []}"
            )
            logger.info(f"  mcp_tools={self.mcp_tools}")

            options = ClaudeCodeOptions(
                max_turns=self.max_turns,
                allowed_tools=allowed_tools,  # Use configured allowed_tools
                permission_mode=self.permission_mode,
                permission_prompt_tool_name=self.permission_prompt_tool,
                mcp_servers=self.mcp_servers,
                mcp_tools=self.mcp_tools,
                system_prompt=system if system else None,
                model=self.model,
            )

            # If we have a streaming handler, use it
            if streaming_handler:
                message_stream = query(prompt=prompt, options=options)
                await streaming_handler.handle_claude_code_stream(message_stream)
                return streaming_handler.get_complete_response()
            else:
                # Fall back to non-streaming collection
                response_parts = []
                async for message in query(prompt=prompt, options=options):
                    if hasattr(message, "content"):
                        if isinstance(message.content, list):
                            for block in message.content:
                                if hasattr(block, "text"):
                                    response_parts.append(block.text)
                                elif isinstance(block, str):
                                    response_parts.append(block)
                        elif isinstance(message.content, str):
                            response_parts.append(message.content)

                return "".join(response_parts)
        except Exception as e:
            logger.error(f"Claude Code streaming generate failed: {e}")
            raise


class MultiLLMCoordinator:
    """Coordinates multiple LLM providers for different tasks."""

    def __init__(self, providers: Dict[str, BaseLLMProvider]):
        self.providers = providers

    def get_provider(self, role: str) -> Optional[BaseLLMProvider]:
        """Get provider for a specific role."""
        # Map roles to providers
        role_mapping = {
            "analyzer": "analyzer",
            "enhancer": "generator",
            "optimizer": "analyzer",
            "synthesizer": "generator",
            "reviewer": "analyzer",
        }

        provider_key = role_mapping.get(role, role)
        return self.providers.get(provider_key)


def create_llm_providers(config: Dict[str, Any]) -> Dict[str, BaseLLMProvider]:
    """Create LLM providers from configuration."""
    providers = {}

    # Handle empty config or mock-only config
    if not config or (len(config) == 1 and "mock" in config):
        providers["mock"] = MockLLMProvider()
        return providers

    # Create providers based on config
    for name, provider_config in config.items():
        provider_type = provider_config.get("type", name)

        if provider_type == "mock" or name == "mock":
            providers[name] = MockLLMProvider()
        elif provider_type == "claude" or (name == "claude" and provider_config.get("api_key")):
            providers[name] = ClaudeLLMProvider(
                api_key=provider_config["api_key"], config=provider_config
            )
        elif provider_type == "claude_code" or name == "claude_code":
            try:
                providers[name] = ClaudeCodeProvider(
                    api_key=provider_config.get("api_key"), config=provider_config
                )
                logger.info("Claude Code provider initialized successfully")
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Failed to initialize Claude Code provider: {e}")
        elif (
            provider_type in ["openai", "lmstudio"]
            or "api_key" in provider_config
            or "base_url" in provider_config
        ):
            providers[name] = OpenAILLMProvider(
                api_key=provider_config.get("api_key", "not-needed"), config=provider_config
            )
        elif provider_type == "local" and provider_config.get("enabled"):
            providers[name] = LocalLLMProvider(config=provider_config)

    # Always include mock provider as fallback
    if "mock" not in providers:
        providers["mock"] = MockLLMProvider()

    return providers
