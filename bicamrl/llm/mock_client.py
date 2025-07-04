"""Mock LLM client for testing."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_logger
from .base_client import BaseLLMClient

logger = get_logger("llm.mock_client")


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without external dependencies."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize mock client."""
        super().__init__(api_key, config)
        self.delay = self.config.get("delay", 0.1)  # Simulate processing time
        logger.info("Initialized mock LLM client")

    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Simulate LLM request with context-aware responses."""
        # Simulate processing delay
        await asyncio.sleep(self.delay)

        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"].lower()
                break

        # Generate context-aware mock responses
        if "world model" in user_message or "analyze this interaction" in user_message:
            return json.dumps(
                {
                    "domain": "software_development",
                    "confidence": 0.8,
                    "entities": [
                        {
                            "id": "file_1",
                            "type": "source_file",
                            "properties": {"path": "test.py", "language": "python"},
                        }
                    ],
                    "relations": [
                        {
                            "source_id": "file_1",
                            "target_id": "user",
                            "type": "edits",
                            "properties": {},
                        }
                    ],
                    "goals": [
                        {"type": "bug_fix", "description": "Fix the issue", "confidence": 0.7}
                    ],
                }
            )

        elif "pattern" in user_message:
            return json.dumps(
                {
                    "patterns": [
                        {
                            "name": "Test Pattern",
                            "type": "workflow",
                            "description": "Test pattern detected",
                            "frequency": 5,
                            "confidence": 0.8,
                            "sequence": ["action1", "action2", "action3"],
                            "trigger_conditions": {"context": "test"},
                            "recommendation": "Consider automating this workflow",
                        }
                    ]
                }
            )

        elif "consolidate" in user_message or "work session" in user_message:
            return json.dumps(
                {
                    "sessions": [
                        {
                            "interactions": [0, 1, 2],
                            "theme": "debugging authentication",
                            "duration_minutes": 30,
                            "key_actions": ["debug", "fix", "test"],
                        }
                    ],
                    "insights": "User frequently debugs authentication issues",
                    "recommendations": ["Add better error messages", "Improve auth documentation"],
                }
            )

        elif "role" in user_message or "behavioral" in user_message:
            return json.dumps(
                {
                    "patterns": [
                        {
                            "pattern_id": "debug_role_001",
                            "domain": "debugging",
                            "recurring_goals": ["fix bugs", "understand errors"],
                            "key_entities": ["debugger", "error_logs"],
                            "action_sequences": [
                                ["read_error", "set_breakpoint", "inspect_vars"],
                                ["check_logs", "trace_execution", "fix_code"],
                            ],
                            "success_indicators": {"bugs_fixed": 5, "time_saved": "2h"},
                            "context_conditions": {"error_present": True, "debugging_mode": True},
                            "frequency": 0.7,
                            "confidence": 0.85,
                        }
                    ]
                }
            )

        elif "improve" in user_message or "enhance" in user_message:
            return "Enhanced: " + messages[-1]["content"][:100] + "..."

        elif "hello" in user_message or "test" in user_message:
            return "Hello! This is a mock response for testing. How can I help you today?"

        else:
            # Default response
            return json.dumps(
                {
                    "analysis": "Mock analysis completed",
                    "confidence": 0.75,
                    "summary": f"Analyzed {len(messages)} messages",
                }
            )

    async def close(self):
        """No resources to close for mock client."""
        pass
