"""Wake agent for chat interface."""

import asyncio
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..core.llm_service import LLMService
from ..core.memory import Memory
from ..sleep.sleep import Sleep
from ..utils.logging_config import get_logger
from .async_bridge import AsyncBridge
from .streaming_handler import StreamingHandler, StreamingUpdate

logger = get_logger("wake_agent")


class WakeAgent:
    """Wake agent that handles user interactions."""

    def __init__(
        self,
        memory: Memory,
        sleep: Optional[Sleep] = None,
        llm_service: Optional[LLMService] = None,
        on_message: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_streaming_update: Optional[Callable[[StreamingUpdate], None]] = None,
    ):
        self.memory = memory
        self.sleep = sleep
        self.llm_service = llm_service
        self.on_message = on_message
        self.on_interrupt = on_interrupt
        self.on_streaming_update = on_streaming_update
        self._async_bridge: Optional[AsyncBridge] = None
        self.session_id = datetime.now().isoformat()
        self._current_task: Optional[asyncio.Task] = None
        self._is_processing = False
        self._was_interrupted = False
        self._streaming_handler: Optional[StreamingHandler] = None
        self._conversation_history: List[Dict[str, str]] = []  # Store conversation history

    async def process_message_async(self, message: str) -> str:
        """Async version of process_message.

        This is the main implementation that uses async services.
        """
        self._is_processing = True
        self._was_interrupted = False
        logger.info(f"Wake agent starting to process: {message}")

        try:
            logger.info(f"Wake processing: {message}")

            if self.llm_service:
                # Create a task for interruptible processing
                self._current_task = asyncio.create_task(self._process_with_llm(message))

                try:
                    response = await self._current_task
                    return response
                except asyncio.CancelledError:
                    logger.info("Message processing was interrupted")
                    self._was_interrupted = True
                    if self.on_interrupt:
                        self.on_interrupt()
                    return "Processing interrupted."
            else:
                # Fallback to mock response if no LLM service
                return f"Hello! This is a mock response for: {message}"

        except Exception as e:
            logger.error(f"Error in Wake: {e}", exc_info=True)
            return f"Error: {str(e)}"
        finally:
            self._is_processing = False
            self._current_task = None

    async def _process_with_llm(self, message: str) -> str:
        """Internal method to process with LLM."""
        # Store user interaction with required fields
        user_interaction = {
            "interaction_id": f"chat_{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "user_query": message,
            "sender": "user",
            "receiver": "assistant",
            "content": message,
            "metadata": {"type": "chat_message"},
            "success": True,
            "feedback_type": "none",
            "tokens_used": 0,
        }
        await self.memory.store_interaction(user_interaction)

        # Check if we're using Claude Code provider for special handling
        provider_name = self.llm_service.default_provider
        provider = self.llm_service.providers.get(provider_name)
        logger.info(
            f"Using provider: {provider_name}, class: {provider.__class__.__name__ if provider else 'None'}"
        )

        if provider and provider.__class__.__name__ == "ClaudeCodeProvider":
            # Use Claude Code with streaming support
            logger.info("Using Claude Code provider with streaming")
            response = await self._generate_with_claude_code(message, provider)
        else:
            # Regular LLM processing
            logger.info("Using regular LLM provider")

            # Build conversation context
            context_parts = ["You are a helpful assistant. Here is the conversation so far:"]
            for hist in self._conversation_history[-10:]:  # Last 10 exchanges
                if hist["role"] == "user":
                    context_parts.append(f"User: {hist['content']}")
                else:
                    context_parts.append(f"Assistant: {hist['content']}")

            context_parts.append(f"\nUser: {message}\nAssistant:")
            prompt = "\n".join(context_parts)

            response = await self.llm_service.generate(prompt)

            # Update history
            self._conversation_history.append({"role": "user", "content": message})
            self._conversation_history.append({"role": "assistant", "content": response})

        # Store assistant response with required fields
        assistant_interaction = {
            "interaction_id": f"chat_response_{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "user_query": message,
            "ai_interpretation": response,
            "sender": "assistant",
            "receiver": "user",
            "content": response,
            "metadata": {"type": "chat_response", "original_message": message},
            "success": True,
            "feedback_type": "none",
            "tokens_used": len(response.split()),  # Simple token estimate
        }
        await self.memory.store_interaction(assistant_interaction)

        # Trigger callback if set
        if self.on_message:
            self.on_message(response, {"original_message": message})

        return response

    async def _generate_with_claude_code(self, message: str, provider) -> str:
        """Generate response using Claude Code provider with streaming support."""
        try:
            # Build conversation context
            context_parts = []
            for hist in self._conversation_history[-10:]:  # Last 10 exchanges
                if hist["role"] == "user":
                    context_parts.append(f"User: {hist['content']}")
                else:
                    context_parts.append(f"Assistant: {hist['content']}")

            # Add current message with context
            if context_parts:
                full_prompt = "\n".join(context_parts) + f"\nUser: {message}\nAssistant:"
            else:
                full_prompt = message

            # Check if we should use streaming
            if self.on_streaming_update and hasattr(provider, "generate_stream"):
                # Use streaming version
                self._streaming_handler = StreamingHandler(self.on_streaming_update)
                response = await provider.generate_stream(
                    full_prompt, system="", streaming_handler=self._streaming_handler
                )
                # Update history
                self._conversation_history.append({"role": "user", "content": message})
                self._conversation_history.append({"role": "assistant", "content": response})
                return response
            else:
                # Fall back to non-streaming version
                response = await provider.generate(full_prompt, system="")
                # Update history
                self._conversation_history.append({"role": "user", "content": message})
                self._conversation_history.append({"role": "assistant", "content": response})
                return response
        except asyncio.CancelledError:
            logger.info("Claude Code generation was cancelled")
            raise
        except Exception as e:
            logger.error(f"Claude Code generation failed: {e}")
            # Fall back to regular generation
            prompt = f"User: {message}\nAssistant:"
            return await self.llm_service.generate(prompt)

    def interrupt(self):
        """Interrupt the current processing."""
        # Check if already interrupted
        if self._was_interrupted:
            return False

        if self._current_task and not self._current_task.done():
            logger.info("Interrupting current Wake processing")
            self._current_task.cancel()
            self._was_interrupted = True
            return True
        return False

    def is_processing(self) -> bool:
        """Check if currently processing a message."""
        return self._is_processing

    def process_message(self, message: str) -> str:
        """Sync wrapper for process_message_async.

        This allows sync code to call the async implementation.
        """
        try:
            # If we have an async bridge, use it
            if self._async_bridge:
                return self._async_bridge.run_async(self.process_message_async(message))
            else:
                # Fallback to creating a new event loop
                # This is less efficient but works
                return asyncio.run(self.process_message_async(message))
        except RuntimeError as e:
            # Handle case where event loop is already running
            if "already running" in str(e):
                logger.warning("Event loop already running, using mock response")
                return f"Hello! This is a mock response for: {message}"
            raise

    def set_async_bridge(self, bridge: AsyncBridge):
        """Set the async bridge for sync/async conversion.

        This should be called by the TUI to provide efficient
        sync/async conversion.
        """
        self._async_bridge = bridge
        logger.info("AsyncBridge set for Wake agent")
