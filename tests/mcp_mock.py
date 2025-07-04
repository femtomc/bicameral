"""Mock MCP types for testing."""

from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Callable
import asyncio

@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mimeType: str
    text: Optional[str] = None

@dataclass
class TextContent:
    text: str
    mimeType: str = "text/plain"

@dataclass
class Tool:
    name: str
    description: str
    inputSchema: Dict[str, Any]

@dataclass
class CallToolResult:
    content: List[TextContent]

@dataclass
class PromptArgument:
    name: str
    description: str
    type: str
    choices: Optional[List[str]] = None
    optional: bool = False

@dataclass
class Prompt:
    name: str
    description: str
    arguments: List[PromptArgument]

@dataclass
class PromptMessage:
    role: str
    content: TextContent

class Server:
    def __init__(self, name: str):
        self.name = name

    def set_list_resources_handler(self, handler):
        pass

    def set_read_resource_handler(self, handler):
        pass

    def set_list_tools_handler(self, handler):
        pass

    def set_call_tool_handler(self, handler):
        pass

    def set_list_prompts_handler(self, handler):
        pass

    def set_run_prompt_handler(self, handler):
        pass

    async def run(self, rx, tx):
        pass

async def stdio_server():
    """Mock stdio server context manager."""
    class MockStdioServer:
        async def __aenter__(self):
            return (None, None)

        async def __aexit__(self, *args):
            pass

    return MockStdioServer()


# Additional mock types for FastMCP
@dataclass
class ImageContent:
    data: bytes
    mimeType: str

@dataclass
class EmbeddedResource:
    uri: str
    content: Any


class FastMCP:
    """Mock FastMCP server for testing."""

    def __init__(self, name: str, version: str, description: str):
        self.name = name
        self.version = version
        self.description = description
        self._tool_handlers: Dict[str, Callable] = {}
        self._resource_handlers: Dict[str, Callable] = {}
        self._on_server_start: Optional[Callable] = None
        self._on_server_stop: Optional[Callable] = None

    def tool(self, name: Optional[str] = None):
        """Decorator for registering tools."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            self._tool_handlers[tool_name] = func
            return func
        return decorator

    def resource(self, uri: str):
        """Decorator for registering resources."""
        def decorator(func: Callable) -> Callable:
            self._resource_handlers[uri] = func
            return func
        return decorator

    def on_server_start(self):
        """Decorator for server start hook."""
        def decorator(func: Callable) -> Callable:
            self._on_server_start = func
            return func
        return decorator

    def on_server_stop(self):
        """Decorator for server stop hook."""
        def decorator(func: Callable) -> Callable:
            self._on_server_stop = func
            return func
        return decorator

    async def run(self):
        """Mock run method."""
        if self._on_server_start:
            await self._on_server_start()
        # Mock server running
        await asyncio.sleep(0.1)
        if self._on_server_stop:
            await self._on_server_stop()
