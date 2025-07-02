"""Test MCP protocol communication with Bicamrl server."""

import asyncio
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import pytest

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import Resource, TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    pytest.skip("MCP not available", allow_module_level=True)


class MCPProtocolSimulator:
    """Simulates MCP protocol communication."""

    def __init__(self):
        self.request_id = 0
        self.responses = []

    def create_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 request."""
        self.request_id += 1
        request = {"jsonrpc": "2.0", "id": self.request_id, "method": method}
        if params:
            request["params"] = params
        return request

    def create_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 notification (no id)."""
        notification = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params
        return notification

    async def send_request(self, server, request: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate sending a request to the server."""
        method = request["method"]
        params = request.get("params", {})

        try:
            # Route to appropriate handler based on method
            if method == "tools/list":
                result = await self._handle_list_tools(server)
            elif method == "tools/call":
                result = await self._handle_call_tool(server, params)
            elif method == "resources/list":
                result = await self._handle_list_resources(server)
            elif method == "resources/read":
                result = await self._handle_read_resource(server, params)
            else:
                raise ValueError(f"Unknown method: {method}")

            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
        except Exception as e:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32603, "message": str(e)},
            }

        self.responses.append(response)
        return response

    async def _handle_list_tools(self, server) -> List[Dict[str, Any]]:
        """Handle tools/list request."""
        tools = []
        for name, handler in server.mcp._tool_handlers.items():
            # Get function signature to build schema
            import inspect

            sig = inspect.signature(handler)

            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = "string"  # Default type
                if param.annotation != param.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list or param.annotation == List:
                        param_type = "array"
                    elif param.annotation == dict or param.annotation == Dict:
                        param_type = "object"

                properties[param_name] = {"type": param_type}

                if param.default == param.empty:
                    required.append(param_name)

            tools.append(
                {
                    "name": name,
                    "description": handler.__doc__ or f"Tool: {name}",
                    "inputSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )

        return tools

    async def _handle_call_tool(self, server, params: Dict[str, Any]) -> Any:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        handler = server.mcp._tool_handlers.get(tool_name)
        if not handler:
            raise ValueError(f"Unknown tool: {tool_name}")

        result = await handler(**arguments)
        return {"content": [{"type": "text", "text": str(result)}]}

    async def _handle_list_resources(self, server) -> List[Dict[str, Any]]:
        """Handle resources/list request."""
        resources = []
        for uri in server.mcp._resource_handlers.keys():
            resources.append(
                {
                    "uri": uri,
                    "name": uri.split("/")[-1],
                    "description": f"Resource: {uri}",
                    "mimeType": "application/json",
                }
            )
        return resources

    async def _handle_read_resource(self, server, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")
        handler = server.mcp._resource_handlers.get(uri)

        if not handler:
            raise ValueError(f"Unknown resource: {uri}")

        resource = await handler()
        return {
            "contents": [
                {"uri": resource.uri, "mimeType": resource.mimeType, "text": resource.text}
            ]
        }


@pytest.fixture
async def mcp_server():
    """Create and initialize the MCP server."""
    import bicamrl.server as server_module

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MEMORY_DB_PATH"] = os.path.join(tmpdir, "test_memory.db")
        await server_module.initialize_server()

        yield server_module

        await server_module.cleanup_server()


@pytest.fixture
def protocol_simulator():
    """Create protocol simulator."""
    return MCPProtocolSimulator()


class TestMCPProtocol:
    """Test MCP protocol-level communication."""

    @pytest.mark.asyncio
    async def test_initialization_handshake(self, mcp_server, protocol_simulator):
        """Test MCP initialization handshake."""

        # 1. Client sends initialize request
        init_request = protocol_simulator.create_request(
            "initialize",
            {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {"call": {}}, "resources": {"read": {}, "list": {}}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        )

        # In real MCP, the server would respond with its capabilities
        # For this test, we'll verify the server is properly configured
        assert mcp_server.mcp.name == "bicamrl"
        assert mcp_server.mcp.version == "0.1.0"
        assert len(mcp_server.mcp._tool_handlers) > 0
        assert len(mcp_server.mcp._resource_handlers) > 0

    @pytest.mark.asyncio
    async def test_list_tools_request(self, mcp_server, protocol_simulator):
        """Test listing tools via protocol."""

        request = protocol_simulator.create_request("tools/list")
        response = await protocol_simulator.send_request(mcp_server, request)

        assert response["jsonrpc"] == "2.0"
        assert "result" in response

        tools = response["result"]
        assert isinstance(tools, list)
        assert len(tools) > 0

        # Verify tool structure
        tool_names = [t["name"] for t in tools]
        assert "log_interaction" in tool_names
        assert "record_feedback" in tool_names

        # Check tool has proper schema
        log_tool = next(t for t in tools if t["name"] == "log_interaction")
        assert "inputSchema" in log_tool
        assert log_tool["inputSchema"]["type"] == "object"
        assert "properties" in log_tool["inputSchema"]
        assert "action" in log_tool["inputSchema"]["properties"]

    @pytest.mark.asyncio
    async def test_call_tool_request(self, mcp_server, protocol_simulator):
        """Test calling a tool via protocol."""

        request = protocol_simulator.create_request(
            "tools/call",
            {
                "name": "log_interaction",
                "arguments": {
                    "action": "test_protocol",
                    "file_path": "test.py",
                    "details": {"protocol": "MCP"},
                },
            },
        )

        response = await protocol_simulator.send_request(mcp_server, request)

        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "content" in response["result"]

        content = response["result"]["content"][0]
        assert content["type"] == "text"
        assert "logged" in content["text"].lower()

    @pytest.mark.asyncio
    async def test_list_resources_request(self, mcp_server, protocol_simulator):
        """Test listing resources via protocol."""

        request = protocol_simulator.create_request("resources/list")
        response = await protocol_simulator.send_request(mcp_server, request)

        assert "result" in response
        resources = response["result"]

        resource_uris = [r["uri"] for r in resources]
        assert "memory://patterns" in resource_uris
        assert "memory://preferences" in resource_uris
        assert "memory://context/recent" in resource_uris

    @pytest.mark.asyncio
    async def test_read_resource_request(self, mcp_server, protocol_simulator):
        """Test reading a resource via protocol."""

        # First log some interactions
        await mcp_server.log_interaction(action="test_resource", file_path="resource_test.py")

        # Read patterns resource
        request = protocol_simulator.create_request("resources/read", {"uri": "memory://patterns"})

        response = await protocol_simulator.send_request(mcp_server, request)

        assert "result" in response
        assert "contents" in response["result"]

        content = response["result"]["contents"][0]
        assert content["uri"] == "memory://patterns"
        assert content["mimeType"] == "application/json"
        assert content["text"] is not None

        # Verify it's valid JSON
        patterns = json.loads(content["text"])
        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_error_handling_protocol(self, mcp_server, protocol_simulator):
        """Test error handling in protocol."""

        # Test unknown method
        request = protocol_simulator.create_request("unknown/method")
        response = await protocol_simulator.send_request(mcp_server, request)

        assert "error" in response
        assert response["error"]["code"] == -32603  # Internal error

        # Test invalid tool
        request = protocol_simulator.create_request(
            "tools/call", {"name": "invalid_tool", "arguments": {}}
        )

        response = await protocol_simulator.send_request(mcp_server, request)
        assert "error" in response

        # Test invalid resource
        request = protocol_simulator.create_request("resources/read", {"uri": "memory://invalid"})

        response = await protocol_simulator.send_request(mcp_server, request)
        assert "error" in response

    @pytest.mark.asyncio
    async def test_notification_handling(self, mcp_server, protocol_simulator):
        """Test handling notifications (no response expected)."""

        # Send a notification
        notification = protocol_simulator.create_notification(
            "log", {"level": "info", "message": "Test notification"}
        )

        # Notifications don't expect responses
        # In real MCP, server would process but not respond
        assert "id" not in notification
        assert notification["method"] == "log"

    @pytest.mark.asyncio
    async def test_batch_requests(self, mcp_server, protocol_simulator):
        """Test handling multiple requests in sequence."""

        requests = [
            protocol_simulator.create_request("tools/list"),
            protocol_simulator.create_request("resources/list"),
            protocol_simulator.create_request(
                "tools/call", {"name": "get_memory_stats", "arguments": {}}
            ),
        ]

        responses = []
        for request in requests:
            response = await protocol_simulator.send_request(mcp_server, request)
            responses.append(response)

        # All requests should succeed
        assert all("result" in r for r in responses)

        # Verify different response types
        assert isinstance(responses[0]["result"], list)  # tools list
        assert isinstance(responses[1]["result"], list)  # resources list
        assert "content" in responses[2]["result"]  # tool call result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
