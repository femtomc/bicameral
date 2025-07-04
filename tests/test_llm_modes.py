"""Test script to demonstrate different LLM modes."""

import pytest
import asyncio
from bicamrl.core.llm_service import LLMService


@pytest.mark.asyncio
@pytest.mark.mock
async def test_with_mock_llm():
    """Test using mock LLM (no external dependencies)."""
    config = {
        "default_provider": "mock",
        "llm_providers": {},
        "rate_limit": 1000
    }

    llm_service = LLMService(config)

    # Test world model inference
    interaction = {
        "user_query": "Help me fix the authentication bug",
        "actions_taken": [
            {"action": "read_file", "file": "auth.py"},
            {"action": "edit_file", "file": "auth.py"},
            {"action": "run_tests", "target": "auth"}
        ],
        "success": True
    }

    response = await llm_service.infer_world_model(interaction)

    assert response.content is not None
    assert response.content['domain'] == 'software_development'
    print(f"Mock response domain: {response.content['domain']}")


@pytest.mark.asyncio
@pytest.mark.lmstudio
async def test_with_lmstudio():
    """Test using LM Studio (requires running server)."""
    config = {
        "default_provider": "lmstudio",
        "llm_providers": {
            "lmstudio": {
                "base_url": "http://localhost:1234/v1",
                "api_key": "not-needed",
                "model": "local-model",
                "max_tokens": 2048,
                "timeout": 60
            }
        },
        "rate_limit": 120
    }

    llm_service = LLMService(config)

    # Test world model inference
    interaction = {
        "user_query": "Help me create a chocolate cake recipe",
        "actions_taken": [
            {"action": "create", "target": "chocolate_cake.md"},
            {"action": "write", "content": "ingredients and steps"}
        ],
        "success": True
    }

    response = await llm_service.infer_world_model(interaction)

    assert response.content is not None
    print(f"\nLM Studio response:")
    print(f"Domain: {response.content.get('domain', 'unknown')}")
    print(f"Confidence: {response.content.get('confidence', 0)}")
    print(f"Raw analysis preview: {response.content.get('raw_analysis', '')[:200]}...")


@pytest.mark.asyncio
async def test_default_behavior(llm_service):
    """Test default behavior (uses fixture which defaults to mock)."""
    # This test will use whatever the fixture provides
    interaction = {
        "user_query": "Compose a melody in C major",
        "actions_taken": [
            {"action": "create", "file": "melody.mid"},
            {"action": "add", "element": "chord_progression"}
        ],
        "success": True
    }

    response = await llm_service.infer_world_model(interaction)
    assert response.content is not None
    print(f"\nDefault fixture response domain: {response.content.get('domain', 'unknown')}")


if __name__ == "__main__":
    # Run specific tests
    print("Running mock test...")
    pytest.main([__file__, "-k", "test_with_mock_llm", "-v", "-s"])

    print("\nRunning LM Studio test (requires server)...")
    pytest.main([__file__, "-k", "test_with_lmstudio", "-v", "-s"])
