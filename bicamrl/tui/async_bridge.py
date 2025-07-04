"""Bridge between sync and async code for TUI."""

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Optional

from ..utils.logging_config import get_logger

logger = get_logger("async_bridge")


class AsyncBridge:
    """Bridge between sync and async code.

    This allows synchronous code (like Rust callbacks) to execute
    async Python code and wait for results.
    """

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self):
        """Start the async event loop in a background thread."""

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self._started.set()
            logger.info("AsyncBridge event loop started")
            self.loop.run_forever()
            logger.info("AsyncBridge event loop stopped")

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        self._started.wait()  # Wait for loop to be ready
        logger.info("AsyncBridge initialized and ready")

    def stop(self):
        """Stop the event loop."""
        if self.loop and self.loop.is_running():
            logger.info("Stopping AsyncBridge event loop")
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
                if self.thread.is_alive():
                    logger.warning("AsyncBridge thread did not stop cleanly")

    def run_async(self, coro) -> Any:
        """Run an async coroutine from sync code and wait for result.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine

        Raises:
            RuntimeError: If the bridge is not started
            Any exception raised by the coroutine
        """
        if not self.loop:
            raise RuntimeError("AsyncBridge not started")

        if not self.loop.is_running():
            raise RuntimeError("AsyncBridge event loop is not running")

        # Schedule the coroutine and wait for result
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            # Block until the coroutine completes
            return future.result()
        except Exception as e:
            logger.error(f"Error running async coroutine: {e}")
            raise

    def run_async_no_wait(self, coro) -> Future:
        """Run an async coroutine without waiting for result.

        Args:
            coro: The coroutine to run

        Returns:
            A Future that will contain the result

        Raises:
            RuntimeError: If the bridge is not started
        """
        if not self.loop:
            raise RuntimeError("AsyncBridge not started")

        if not self.loop.is_running():
            raise RuntimeError("AsyncBridge event loop is not running")

        # Schedule the coroutine but don't wait
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
