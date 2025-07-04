"""Test LM Studio connection and configuration."""

import pytest
import os
from bicamrl.sleep.llm_providers import OpenAILLMProvider


@pytest.mark.lmstudio
@pytest.mark.asyncio
async def test_lmstudio_connection():
    """Test that we can connect to LM Studio."""
    config = {
        "base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "api_key": "not-needed",
        "model": os.getenv("LMSTUDIO_MODEL", "local-model"),
        "max_tokens": 100
    }

    provider = OpenAILLMProvider(api_key="not-needed", config=config)

    # Test simple generation
    response = await provider.generate("Hello, this is a test. Please respond with 'OK'.")
    assert response is not None
    assert len(response) > 0
    print(f"LM Studio response: {response}")


@pytest.mark.lmstudio
@pytest.mark.asyncio
async def test_lmstudio_analysis():
    """Test LM Studio can handle analysis tasks."""
    config = {
        "base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "api_key": "not-needed",
        "model": os.getenv("LMSTUDIO_MODEL", "local-model"),
        "max_tokens": 500
    }

    provider = OpenAILLMProvider(api_key="not-needed", config=config)

    # Test analysis (generate instead of analyze to avoid JSON requirement)
    prompt = """Analyze this user interaction:
User: "Help me fix the login bug"
Actions: read auth.py, edit auth.py, run tests
Success: True

What domain is this? Return a single word."""

    response = await provider.generate(prompt)
    assert response is not None
    print(f"Analysis response: {response}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_lmstudio_connection.py -v -s
    pytest.main([__file__, "-v", "-s"])
