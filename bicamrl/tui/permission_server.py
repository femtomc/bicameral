"""MCP server for handling tool permissions in the TUI."""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import aiohttp
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# When running as MCP server, minimize logging to avoid cluttering Claude's output
if os.environ.get("MCP_MODE") or (len(sys.argv) > 0 and "permission_server" in sys.argv[0]):
    # Running as MCP server - only show warnings and errors
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logger = logging.getLogger("permission_server")
    # Also suppress aiohttp logs
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
else:
    # Running in test/debug mode - use normal logging
    from ..utils.logging_config import get_logger

    logger = get_logger("permission_server")


class PermissionResponse(Enum):
    """Permission response types."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class ToolPermissionRequest:
    """Request for tool permission."""

    tool_name: str
    tool_input: Dict[str, Any]
    context: Optional[str] = None


class PermissionManager:
    """Manages tool permissions with configurable policies."""

    def __init__(self):
        # Start with empty lists - everything requires permission
        self.always_allowed = set()  # Empty - nothing is auto-allowed

        self.always_denied = set()  # Empty - nothing is auto-denied

        # Everything will go through ask_user by default
        self.ask_user = set()

        # Callback for user prompts (set by TUI)
        self.user_prompt_callback: Optional[Callable] = None

    async def check_permission(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> PermissionResponse:
        """Check if a tool should be allowed to run."""

        # Check always allowed
        if tool_name in self.always_allowed:
            logger.info(f"Tool {tool_name} is always allowed")
            return PermissionResponse.ALLOW

        # Check always denied
        if tool_name in self.always_denied:
            logger.warning(f"Tool {tool_name} is always denied")
            return PermissionResponse.DENY

        # Check if we should ask the user
        if tool_name in self.ask_user:
            logger.info(f"Tool {tool_name} requires user permission")
            return PermissionResponse.ASK

        # Default to asking for unknown tools
        logger.info(f"Unknown tool {tool_name}, asking user")
        return PermissionResponse.ASK


# Global permission manager instance
_permission_manager: Optional[PermissionManager] = None

# Create the FastMCP instance
mcp = FastMCP("bicamrl-permissions")


def setup_permission_tools(permission_manager: PermissionManager):
    """Set up permission tools for the MCP server."""
    global _permission_manager
    _permission_manager = permission_manager

    @mcp.tool()
    async def approval_prompt(tool_name: str, input: Dict[str, Any]) -> TextContent:
        """Handle permission request for a tool.

        This tool is called by Claude Code when it needs permission to use a tool.
        """
        logger.info(f"Permission request for tool: {tool_name}")
        logger.debug(f"Tool input: {input}")

        try:
            # Check permission policy
            permission = await _permission_manager.check_permission(tool_name, input)

            if permission == PermissionResponse.ALLOW:
                # Automatically allow
                response = {"behavior": "allow", "updatedInput": input}
                logger.info(f"Automatically allowed {tool_name}")

            elif permission == PermissionResponse.DENY:
                # Automatically deny
                response = {
                    "behavior": "deny",
                    "message": f"Tool {tool_name} is not allowed by security policy",
                }
                logger.warning(f"Automatically denied {tool_name}")

            else:  # ASK
                # Ask the user via HTTP request to TUI
                tui_url = os.environ.get("BICAMRL_TUI_URL", "http://localhost:8766")

                try:
                    async with aiohttp.ClientSession() as session:
                        # Send permission request to TUI
                        request_data = {
                            "tool_name": tool_name,
                            "tool_input": input,
                            "request_id": f"{tool_name}_{asyncio.get_event_loop().time()}",
                        }

                        logger.info(f"Sending permission request to TUI at {tui_url}")
                        logger.info(f"Request data: {request_data}")
                        async with session.post(
                            f"{tui_url}/permission",
                            json=request_data,
                            timeout=aiohttp.ClientTimeout(total=60),
                        ) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                user_allowed = result.get("allowed", False)

                                if user_allowed:
                                    response = {"behavior": "allow", "updatedInput": input}
                                    logger.info(f"User allowed {tool_name}")

                                    # Check if user wants to update policy
                                    if result.get("always_allow"):
                                        _permission_manager.always_allowed.add(tool_name)
                                        _permission_manager.ask_user.discard(tool_name)
                                else:
                                    response = {
                                        "behavior": "deny",
                                        "message": f"User denied permission for {tool_name}",
                                    }
                                    logger.info(f"User denied {tool_name}")

                                    # Check if user wants to update policy
                                    if result.get("always_deny"):
                                        _permission_manager.always_denied.add(tool_name)
                                        _permission_manager.ask_user.discard(tool_name)
                            else:
                                # TUI returned error
                                response = {
                                    "behavior": "deny",
                                    "message": f"Permission request failed: HTTP {resp.status}",
                                }
                                logger.error(f"TUI returned status {resp.status}")

                except asyncio.TimeoutError:
                    response = {"behavior": "deny", "message": "Permission request timed out"}
                    logger.error("Permission request to TUI timed out")
                except Exception as e:
                    response = {"behavior": "deny", "message": f"Failed to contact TUI: {str(e)}"}
                    logger.error(f"Failed to contact TUI: {e}")

            # Return JSON string as required by Claude Code
            return TextContent(type="text", text=json.dumps(response))

        except Exception as e:
            logger.error(f"Error in permission prompt: {e}")
            # On error, deny the request
            return TextContent(
                type="text",
                text=json.dumps(
                    {"behavior": "deny", "message": f"Permission check failed: {str(e)}"}
                ),
            )

    @mcp.tool()
    async def update_permission_policy(
        tool_name: str,
        policy: str,  # "allow", "deny", or "ask"
    ) -> TextContent:
        """Update the permission policy for a tool."""

        if policy == "allow":
            _permission_manager.always_allowed.add(tool_name)
            _permission_manager.always_denied.discard(tool_name)
            _permission_manager.ask_user.discard(tool_name)
            message = f"Tool {tool_name} will now be automatically allowed"

        elif policy == "deny":
            _permission_manager.always_denied.add(tool_name)
            _permission_manager.always_allowed.discard(tool_name)
            _permission_manager.ask_user.discard(tool_name)
            message = f"Tool {tool_name} will now be automatically denied"

        elif policy == "ask":
            _permission_manager.ask_user.add(tool_name)
            _permission_manager.always_allowed.discard(tool_name)
            _permission_manager.always_denied.discard(tool_name)
            message = f"Tool {tool_name} will now require user permission"

        else:
            message = f"Invalid policy: {policy}. Use 'allow', 'deny', or 'ask'"

        logger.info(message)
        return TextContent(type="text", text=message)

    @mcp.tool()
    async def list_permission_policies() -> TextContent:
        """List current permission policies for all tools."""

        policies = {
            "always_allowed": list(_permission_manager.always_allowed),
            "always_denied": list(_permission_manager.always_denied),
            "ask_user": list(_permission_manager.ask_user),
        }

        return TextContent(type="text", text=json.dumps(policies, indent=2))


async def run_permission_server(permission_manager: PermissionManager):
    """Run the permission server as a stdio server."""
    # Set up the tools with the permission manager
    setup_permission_tools(permission_manager)

    # Run as stdio server (communicates via stdin/stdout)
    await mcp.run_stdio_async()


if __name__ == "__main__":
    # For testing
    manager = PermissionManager()
    asyncio.run(run_permission_server(manager))
