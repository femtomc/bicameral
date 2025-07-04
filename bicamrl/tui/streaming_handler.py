"""Streaming support for TUI real-time updates."""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

from ..utils.logging_config import get_logger

logger = get_logger("streaming_handler")


class StreamingMessageType(Enum):
    """Types of streaming messages."""

    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    USAGE = "usage"
    ERROR = "error"


@dataclass
class StreamingUpdate:
    """Represents a streaming update to send to the UI."""

    type: StreamingMessageType
    content: str
    metadata: Dict[str, Any]
    is_complete: bool = False


class StreamingHandler:
    """Handles streaming responses from LLMs and formats them for the TUI."""

    def __init__(self, on_update: Optional[Callable[[StreamingUpdate], None]] = None):
        self.on_update = on_update
        self.current_response = []
        self.token_count = 0
        self.tool_uses = []
        self.approximate_tokens = 0

    async def handle_claude_code_stream(self, message_stream):
        """Handle streaming messages from Claude Code SDK."""
        try:
            async for message in message_stream:
                # Handle different message types
                if hasattr(message, "__class__"):
                    message_type = message.__class__.__name__
                    # Log important message types
                    if message_type in ["ToolResult", "ToolUseBlock", "ToolResultBlock"]:
                        logger.info(f"[STREAMING] Received message type: {message_type}")

                    if message_type == "AssistantMessage":
                        await self._handle_assistant_message(message)
                    elif message_type == "SystemMessage":
                        await self._handle_system_message(message)
                    elif message_type == "ResultMessage":
                        await self._handle_result_message(message)
                    elif message_type == "UserMessage":
                        # User messages in stream could be permission prompts
                        await self._handle_user_message(message)
                    elif message_type == "ToolPermissionRequest":
                        # Handle tool permission requests
                        await self._handle_tool_permission_request(message)
                    elif message_type == "ToolResult":
                        # Handle tool results at the top level
                        logger.info(f"Top-level ToolResult message received")
                        # Send close popup event
                        if self.on_update:
                            self.on_update(
                                StreamingUpdate(
                                    type=StreamingMessageType.SYSTEM,
                                    content="__CLOSE_POPUP__",
                                    metadata={"close_popup": True},
                                )
                            )
                    else:
                        # Unknown message type - only log if it's really unexpected
                        if message_type not in ["TextContent"]:
                            logger.debug(f"Unhandled message type: {message_type}")

        except asyncio.CancelledError:
            # Handle interruption gracefully
            if self.on_update:
                self.on_update(
                    StreamingUpdate(
                        type=StreamingMessageType.SYSTEM,
                        content="Response interrupted",
                        metadata={"interrupted": True},
                        is_complete=True,
                    )
                )
            raise

    async def _handle_assistant_message(self, message):
        """Handle assistant message with content blocks."""
        if hasattr(message, "content") and isinstance(message.content, list):
            for block in message.content:
                block_type = block.__class__.__name__

                if block_type == "TextBlock":
                    # Regular text content
                    self.current_response.append(block.text)

                    # Better token estimation based on character count
                    # Claude models average ~3.5 chars per token
                    token_increment = max(1, int(len(block.text) / 3.5))
                    self.approximate_tokens += token_increment

                    # Send token update
                    if self.on_update:
                        self.on_update(
                            StreamingUpdate(
                                type=StreamingMessageType.SYSTEM,
                                content=f"__UPDATE_TOKENS__:{self.approximate_tokens}",
                                metadata={"token_count": self.approximate_tokens},
                                is_complete=False,
                            )
                        )

                        self.on_update(
                            StreamingUpdate(
                                type=StreamingMessageType.TEXT,
                                content=block.text,
                                metadata={"partial": True},
                            )
                        )

                elif block_type == "ToolUseBlock":
                    # Tool being used - this should trigger permission if needed
                    self.tool_uses.append(
                        {"id": block.id, "name": block.name, "input": block.input}
                    )
                    logger.info(f"Tool use detected: {block.name} with id: {block.id}")

                    # First show that we're using the tool
                    if self.on_update:
                        self.on_update(
                            StreamingUpdate(
                                type=StreamingMessageType.TOOL_USE,
                                content=f"Using tool: {block.name}",
                                metadata={
                                    "tool_id": block.id,
                                    "tool_name": block.name,
                                    "tool_input": block.input,
                                },
                            )
                        )

                elif block_type == "ToolResultBlock":
                    # Tool result - close the tool popup
                    logger.info(
                        f"Tool result received for id: {getattr(block, 'tool_use_id', 'unknown')}"
                    )
                    if self.on_update:
                        content = (
                            block.content if isinstance(block.content, str) else str(block.content)
                        )
                        # First send the tool result
                        self.on_update(
                            StreamingUpdate(
                                type=StreamingMessageType.TOOL_RESULT,
                                content=content[:100] + "..." if len(content) > 100 else content,
                                metadata={
                                    "tool_use_id": getattr(block, "tool_use_id", None),
                                    "is_error": getattr(block, "is_error", False),
                                },
                            )
                        )
                        # Then send a close popup event
                        logger.info("Sending __CLOSE_POPUP__ to close tool popup")
                        self.on_update(
                            StreamingUpdate(
                                type=StreamingMessageType.SYSTEM,
                                content="__CLOSE_POPUP__",
                                metadata={"close_popup": True},
                            )
                        )

    async def _handle_system_message(self, message):
        """Handle system messages with metadata."""
        if self.on_update:
            # Format system messages for display based on subtype
            content = ""

            # Handle different system message subtypes
            if message.subtype == "auto_accept":
                content = "Auto-accept mode: ON - Edits will be applied automatically"
            elif message.subtype == "usage_limit":
                # Extract usage details from data
                limit_info = message.data.get("limit_info", {})
                remaining = limit_info.get("remaining_tokens", "unknown")
                total = limit_info.get("total_tokens", "unknown")
                content = f"WARNING: Approaching usage limit: {remaining}/{total} tokens remaining"
            elif message.subtype == "permission_mode":
                mode = message.data.get("mode", "unknown")
                content = f"Permission mode: {mode}"
            else:
                # Default formatting
                content = f"System: {message.subtype}"
                if message.data and "message" in message.data:
                    content = message.data["message"]

            self.on_update(
                StreamingUpdate(
                    type=StreamingMessageType.SYSTEM, content=content, metadata=message.data
                )
            )

    async def _handle_result_message(self, message):
        """Handle result message with usage information."""
        if self.on_update:
            # Extract usage information
            usage_info = {
                "duration_ms": message.duration_ms,
                "duration_api_ms": message.duration_api_ms,
                "total_cost_usd": message.total_cost_usd,
                "usage": message.usage,
            }

            # Format usage for display
            input_tokens = message.usage.get("input_tokens", 0)
            output_tokens = message.usage.get("output_tokens", 0)
            tokens_used = input_tokens + output_tokens
            cost_str = f"${message.total_cost_usd:.4f}" if message.total_cost_usd else "N/A"

            # Send final accurate token count update
            self.on_update(
                StreamingUpdate(
                    type=StreamingMessageType.SYSTEM,
                    content=f"__UPDATE_TOKENS__:{tokens_used}",
                    metadata={"token_count": tokens_used, "final": True},
                    is_complete=False,
                )
            )

            self.on_update(
                StreamingUpdate(
                    type=StreamingMessageType.USAGE,
                    content=f"{input_tokens} in + {output_tokens} out = {tokens_used} tokens | Cost: {cost_str} | Time: {message.duration_ms}ms",
                    metadata=usage_info,
                    is_complete=True,
                )
            )

    def get_complete_response(self) -> str:
        """Get the complete response text."""
        return "".join(self.current_response)

    def reset(self):
        """Reset the handler for a new response."""
        self.current_response = []
        self.token_count = 0
        self.tool_uses = []

    async def _handle_user_message(self, message):
        """Handle user messages that might be permission prompts."""
        # Don't log every user message - too noisy
        if hasattr(message, "content"):
            content = message.content if isinstance(message.content, str) else str(message.content)

            # Check if this is a permission prompt from Claude Code
            # Claude Code sends permission prompts as user messages
            if (
                "permission" in content.lower()
                or "allow" in content.lower()
                or "tool" in content.lower()
            ):
                logger.debug("Detected potential permission prompt")
                # Don't send these as permission requests - Claude Code handles them differently

    async def _handle_tool_permission_request(self, message):
        """Handle explicit tool permission requests."""
        if self.on_update:
            tool_name = getattr(message, "tool_name", "unknown")
            tool_input = getattr(message, "tool_input", "")

            self.on_update(
                StreamingUpdate(
                    type=StreamingMessageType.SYSTEM,
                    content=f"__TOOL_PERMISSION_REQUEST__:{tool_name}:{tool_input}",
                    metadata={
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "permission_request": True,
                    },
                )
            )
