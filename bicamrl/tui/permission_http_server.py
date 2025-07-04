"""HTTP server for handling permission requests from the MCP server."""

import asyncio
import logging
from typing import Callable, Dict, Optional

from aiohttp import web

logger = logging.getLogger("permission_http")


class PermissionHTTPServer:
    """HTTP server that receives permission requests and shows popups in TUI."""

    def __init__(self, port: int = 0):  # Use 0 to get a random available port
        self.port = port
        self.actual_port = None  # Will be set after starting
        self.app = web.Application()
        self.app.router.add_post("/permission", self.handle_permission_request)
        self.app.router.add_get("/health", self.handle_health)

        # Callback to show popup in TUI
        self.show_popup_callback: Optional[Callable] = None

        # Store pending requests
        self.pending_requests: Dict[str, asyncio.Future] = {}

        self.runner = None
        self.site = None

    async def start(self):
        """Start the HTTP server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "localhost", self.port)
        await self.site.start()
        # Get the actual port if we used 0
        if self.port == 0:
            self.actual_port = self.site._server.sockets[0].getsockname()[1]
        else:
            self.actual_port = self.port
        logger.info(f"Permission HTTP server started on http://localhost:{self.actual_port}")
        # Set environment variable for the permission server to find us
        import os

        os.environ["BICAMRL_TUI_URL"] = f"http://localhost:{self.actual_port}"

    async def stop(self):
        """Stop the HTTP server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Permission HTTP server stopped")

    async def handle_health(self, request):
        """Health check endpoint."""
        return web.json_response({"status": "ok"})

    async def handle_permission_request(self, request):
        """Handle permission request from MCP server."""
        try:
            data = await request.json()
            tool_name = data.get("tool_name")
            tool_input = data.get("tool_input")
            request_id = data.get("request_id")

            logger.info(f"Received permission request for {tool_name}")

            if not self.show_popup_callback:
                return web.json_response(
                    {"allowed": False, "error": "No UI callback set"}, status=500
                )

            # Create a future to wait for user response
            future = asyncio.Future()
            self.pending_requests[request_id] = future

            # Show popup in TUI (this should be called in the main thread)
            await self.show_popup_callback(tool_name, tool_input, request_id)

            # Wait for user response (with timeout)
            try:
                result = await asyncio.wait_for(future, timeout=60.0)
                return web.json_response(result)
            except asyncio.TimeoutError:
                logger.warning(f"Permission request {request_id} timed out")
                return web.json_response(
                    {"allowed": False, "error": "Request timed out"}, status=408
                )
            finally:
                # Clean up
                self.pending_requests.pop(request_id, None)

        except Exception as e:
            logger.error(f"Error handling permission request: {e}")
            return web.json_response({"allowed": False, "error": str(e)}, status=500)

    def respond_to_request(
        self, request_id: str, allowed: bool, always_allow: bool = False, always_deny: bool = False
    ):
        """Respond to a pending permission request."""
        if request_id in self.pending_requests:
            future = self.pending_requests[request_id]
            if not future.done():
                future.set_result(
                    {"allowed": allowed, "always_allow": always_allow, "always_deny": always_deny}
                )
                logger.info(f"Responded to request {request_id}: allowed={allowed}")
        else:
            logger.warning(f"No pending request found for {request_id}")
