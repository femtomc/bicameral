"""Python wrapper for the Rust-based TUI using Ratatui."""

import asyncio
import json
import queue
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.llm_service import LLMService
from ..core.memory import Memory
from ..sleep.sleep import Sleep
from ..tui.async_bridge import AsyncBridge
from ..tui.permission_http_server import PermissionHTTPServer
from ..tui.streaming_handler import StreamingUpdate
from ..tui.wake_agent import WakeAgent
from ..utils.logging_config import get_logger

# Import the Rust extension module (will be available after building)
try:
    from ..bicamrl_tui import BicamrlTUI as RustTUI
    from ..bicamrl_tui import Stats
except ImportError:
    # For development, provide a mock until the Rust module is built
    class RustTUI:
        def __init__(self):
            pass

        def run(self):
            print("Rust TUI not yet built. Run 'pixi run build-rust' first.")

        def update_stats(self, stats):
            pass

        def add_message(self, message, msg_type):
            pass

        def refresh(self):
            pass

        def quit(self):
            pass

    class Stats:
        def __init__(self):
            self.total_interactions = 0
            self.total_patterns = 0
            self.success_rate = 0.0
            self.active_memories = 0
            self.tokens_used = 0
            self.active_sessions = 0


logger = get_logger("rust_tui")


class BicamrlRustTUI:
    """High-level Python wrapper for the Rust TUI."""

    def __init__(self, db_path: str = None, config: Optional[Dict[str, Any]] = None):
        self.db_path = db_path or str(Path.home() / ".bicamrl" / "memory")
        self.config = config or {}
        self.memory: Optional[Memory] = None
        self.sleep: Optional[Sleep] = None
        self.wake_agent: Optional[WakeAgent] = None
        self.llm_service: Optional[LLMService] = None
        self.async_bridge: Optional[AsyncBridge] = None

        # Create Rust TUI instance
        self.tui = RustTUI()

        # Set up the callback for messages from Rust
        self.tui.set_callback(self._on_chat_message)

        # Queue for streaming updates to avoid direct calls
        self._streaming_queue = queue.Queue()

        # Permission handling
        self._permission_response_queue = queue.Queue()  # Use regular queue, not asyncio
        self._pending_permission_event = None
        self._permission_manager = None
        self._permission_http_server = None
        self._pending_permission_requests = {}  # Map request_id to tool_name

    async def initialize(self):
        """Initialize the memory and sleep systems."""
        try:
            # Initialize LLM service FIRST based on config
            if self.config.get("llm_providers"):
                self.llm_service = LLMService(self.config)
                logger.info("LLM service initialized from config")
            else:
                # Simple mock LLM service
                self.llm_service = LLMService(
                    {"llm_providers": {"mock": {}}, "default_provider": "mock"}
                )
                logger.info("LLM service initialized with mock provider")

            # Initialize memory with LLM service
            self.memory = Memory(self.db_path, llm_service=self.llm_service)
            logger.info("Memory system initialized")

            # Initialize Wake agent with streaming support
            self.wake_agent = WakeAgent(
                memory=self.memory,
                llm_service=self.llm_service,
                on_streaming_update=self._on_streaming_update,
            )

            # Set the async bridge for efficient sync/async conversion
            if self.async_bridge:
                self.wake_agent.set_async_bridge(self.async_bridge)

            logger.info("Wake agent initialized")

            # Initialize permission HTTP server
            self._permission_http_server = PermissionHTTPServer()
            self._permission_http_server.show_popup_callback = self._show_permission_popup
            await self._permission_http_server.start()
            logger.info("Permission HTTP server started")

            # Add welcome message
            self.tui.add_message("Welcome to Bicamrl! Type a message and press Enter.", "assistant")

        except Exception as e:
            logger.error(f"Failed to initialize systems: {e}")
            raise

    async def _get_current_stats(self) -> Stats:
        """Get current statistics from the memory system."""
        stats = Stats()

        if self.memory:
            memory_stats = await self.memory.get_stats()
            stats.total_interactions = memory_stats.get("total_interactions", 0)
            stats.total_patterns = memory_stats.get("total_patterns", 0)
            stats.active_sessions = memory_stats.get("active_sessions", 0)

            # Calculate success rate from recent interactions
            recent = await self.memory.store.get_complete_interactions(limit=100)
            if recent:
                success_count = sum(1 for i in recent if i.get("success", False))
                stats.success_rate = success_count / len(recent)
                stats.active_memories = len(recent)
                stats.tokens_used = sum(i.get("tokens_used", 0) for i in recent)

        return stats

    def _on_wake_message(self, response: str, metadata: Dict[str, Any]):
        """Handle Wake agent messages."""
        # Update metrics if available
        if "latency" in metadata:
            self._sleep_metrics["llm_latency"] = metadata["latency"]
        if "role" in metadata:
            self._sleep_metrics["active_role"] = metadata.get("role", "None")

    def _on_streaming_update(self, update: StreamingUpdate):
        """Handle streaming updates from Wake agent."""
        # Queue the update instead of calling Rust directly
        # This avoids borrowing issues by letting the main thread handle it
        self._streaming_queue.put(
            {
                "type": update.type.value,
                "content": update.content,
                "metadata": update.metadata,
                "is_complete": update.is_complete,
            }
        )

    def _on_chat_message(self, message: str) -> str:
        """Callback from Rust TUI when user sends a message.

        IMPORTANT: This is called FROM Rust, so we CANNOT call back into Rust!
        We just return the response and let Rust handle the UI update.
        """
        logger.info(f"Received message from TUI: {message}")

        # Handle special polling command for streaming updates
        if message == "__POLL_STREAMING__":
            # Check if there are queued streaming updates
            try:
                update = self._streaming_queue.get_nowait()
                # Handle special thinking state changes
                if update.get("type") == "__START_THINKING__":
                    logger.info("Returning START_THINKING to Rust")
                    return "__START_THINKING__"
                elif update.get("type") == "__STOP_THINKING__":
                    logger.info("Returning STOP_THINKING to Rust")
                    return "__STOP_THINKING__"
                else:
                    # Return as JSON so Rust can parse it
                    # ensure_ascii=False to preserve Unicode characters
                    return f"__STREAMING__:{json.dumps(update, ensure_ascii=False)}"
            except queue.Empty:
                return "__NO_STREAMING__"

        # Handle special commands
        if message == "__INTERRUPT__":
            if self.wake_agent and self.wake_agent.interrupt():
                return "__INTERRUPTED__"
            else:
                return "__NO_ACTIVE_TASK__"

        # Handle tool permission responses
        if message.startswith("__TOOL_PERMISSION__:"):
            # Parse format: __TOOL_PERMISSION__:tool_name:action
            parts = message.split(":", 2)
            if len(parts) == 3:
                _, tool_name, action = parts

                # Look up the request_id using our reverse mapping
                request_id = self._pending_permission_requests.get(tool_name)

                if request_id and self._permission_http_server:
                    logger.info(
                        f"Processing permission response for {tool_name} (request_id: {request_id}): {action}"
                    )

                    # Respond via HTTP server
                    allowed = action in ["allow", "always_allow"]
                    always_allow = action == "always_allow"
                    always_deny = action == "always_deny"

                    self._permission_http_server.respond_to_request(
                        request_id, allowed, always_allow, always_deny
                    )

                    # Clean up both mappings
                    if request_id in self._pending_permission_requests:
                        del self._pending_permission_requests[request_id]
                    if tool_name in self._pending_permission_requests:
                        del self._pending_permission_requests[tool_name]

                    return "__PERMISSION_HANDLED__"
                else:
                    logger.warning(f"No pending request found for tool {tool_name}")
                    return "__NO_PENDING_REQUEST__"
            return "__INVALID_PERMISSION_FORMAT__"

        # For normal messages, start async processing and return immediately
        # This prevents blocking the callback thread
        if self.wake_agent and self.async_bridge:
            # Queue the message for async processing
            async def process_and_queue():
                try:
                    response = await self.wake_agent.process_message_async(message)
                    # If we're NOT using streaming (no Claude Code), queue the response
                    provider_name = self.wake_agent.llm_service.default_provider
                    provider = self.wake_agent.llm_service.providers.get(provider_name)
                    if not (provider and provider.__class__.__name__ == "ClaudeCodeProvider"):
                        # Not Claude Code, so we need to queue the response
                        self._streaming_queue.put(
                            {
                                "type": "text",
                                "content": response,
                                "metadata": {},
                                "is_complete": True,
                            }
                        )
                    # For Claude Code, the response has already been streamed

                    # Stop thinking
                    self._streaming_queue.put(
                        {
                            "type": "__STOP_THINKING__",
                            "content": "",
                            "metadata": {},
                            "is_complete": True,
                        }
                    )
                    logger.info("Queued STOP_THINKING")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self._streaming_queue.put(
                        {
                            "type": "text",
                            "content": f"Error: {str(e)}",
                            "metadata": {"error": True},
                            "is_complete": True,
                        }
                    )

            # Signal that we're starting to think FIRST
            self._streaming_queue.put(
                {"type": "__START_THINKING__", "content": "", "metadata": {}, "is_complete": False}
            )
            logger.info("Queued START_THINKING")

            # Start the async processing
            self.async_bridge.run_async_no_wait(process_and_queue())

            # Return immediately with empty response
            # The actual response will come through streaming
            return "__PROCESSING__"
        else:
            return "Wake agent not initialized"

    async def _show_permission_popup(
        self, tool_name: str, tool_input: Dict[str, Any], request_id: str
    ):
        """Show permission popup in TUI for HTTP server request."""
        logger.info(f"Showing permission popup for tool: {tool_name}, request: {request_id}")

        # Store the request mapping - both directions for easy lookup
        self._pending_permission_requests[request_id] = tool_name
        self._pending_permission_requests[tool_name] = request_id  # Also store reverse mapping

        # Show popup in TUI by sending a special streaming update
        self._streaming_queue.put(
            {
                "type": "system",
                "content": f"__TOOL_PERMISSION_REQUEST__:{tool_name}:{json.dumps(tool_input)}",
                "metadata": {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "permission_request": True,
                    "request_id": request_id,
                },
                "is_complete": False,
            }
        )

    def run(self):
        """Run the TUI."""
        # Create and start the async bridge
        self.async_bridge = AsyncBridge()
        self.async_bridge.start()
        logger.info("AsyncBridge started for TUI")

        try:
            # Initialize everything using the bridge
            self.async_bridge.run_async(self.initialize())

            # Just run the TUI - handle everything in callbacks
            self.tui.run()
        finally:
            # Clean up the async bridge
            if self.async_bridge:
                self.async_bridge.stop()
                logger.info("AsyncBridge stopped")

            # Clean up HTTP server
            if self._permission_http_server:
                asyncio.run(self._permission_http_server.stop())
                logger.info("Permission HTTP server stopped")

    def add_message(self, message: str, msg_type: str = "system"):
        """Add a message to the chat interface."""
        self.tui.add_message(message, msg_type)

    def refresh(self):
        """Refresh the TUI display."""
        self.tui.refresh()


def run_rust_tui(db_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """Run the Rust-based TUI."""
    tui = BicamrlRustTUI(db_path, config)
    tui.run()
