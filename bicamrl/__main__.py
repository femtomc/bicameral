#!/usr/bin/env python3
"""Main entry point for Bicamrl MCP Server."""

import sys

from .server import mcp

if __name__ == "__main__":
    # For testing/direct execution
    if "--test" in sys.argv:
        print("Bicamrl MCP Server - Test Mode")
        print(f"Tools: {len(mcp.tools)}")
        print(f"Resources: {len(mcp.resources)}")
    else:
        mcp.run()
