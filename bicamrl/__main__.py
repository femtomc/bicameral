#!/usr/bin/env python3
"""Main entry point for Bicamrl MCP Server."""

import sys

from .server import mcp


def main():
    """Main entry point that routes to CLI or server."""
    # Check if we should run CLI or server
    if len(sys.argv) > 1 and sys.argv[1] not in ["--test"]:
        # Run CLI for commands like 'dashboard', 'init', etc.
        from .cli import cli

        cli()
    else:
        # Run as MCP server
        if "--test" in sys.argv:
            print("Bicamrl MCP Server - Test Mode")
            print(f"Tools: {len(mcp.tools)}")
            print(f"Resources: {len(mcp.resources)}")
        else:
            mcp.run()


if __name__ == "__main__":
    main()
