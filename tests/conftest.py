"""Pytest configuration and fixtures."""

import pytest
import asyncio
import tempfile
import shutil
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, Any
from datetime import datetime

from bicamrl.core.memory import Memory
from bicamrl.core.feedback_processor import FeedbackProcessor
from bicamrl.core.llm_service import LLMService
from bicamrl.sleep.llm_providers import OpenAILLMProvider


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def temp_dir():
    """Create a temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def lmstudio_config() -> Dict[str, Any]:
    """Configuration for LM Studio testing."""
    return {
        "default_provider": "lmstudio",
        "llm_providers": {
            "lmstudio": {
                "base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                "api_key": "not-needed",  # LM Studio doesn't require API key
                "model": os.getenv("LMSTUDIO_MODEL", "local-model"),
                "max_tokens": 2048,
                "timeout": 60  # Local models can be slower
            }
        },
        "rate_limit": 120  # Higher rate limit for local testing
    }


@pytest.fixture
def mock_llm_config() -> Dict[str, Any]:
    """Configuration for mock LLM (no external dependencies)."""
    return {
        "default_provider": "mock",
        "llm_providers": {},
        "rate_limit": 1000  # No real rate limit for mocks
    }


@pytest.fixture
async def llm_service(request):
    """Create LLM service based on test requirements.
    
    By default uses LM Studio if available, falls back to mock.
    Tests can request specific config with:
    - @pytest.mark.lmstudio - force LM Studio
    - @pytest.mark.mock - force mock
    - @pytest.mark.openai - use OpenAI (requires API key)
    """
    # Check markers
    force_lmstudio = False
    if hasattr(request, 'node'):
        if request.node.get_closest_marker('mock'):
            config = request.getfixturevalue('mock_llm_config')
        elif request.node.get_closest_marker('openai'):
            config = {
                "default_provider": "openai",
                "llm_providers": {
                    "openai": {
                        "api_key": os.getenv("OPENAI_API_KEY"),
                        "model": "gpt-4-turbo-preview"
                    }
                }
            }
        elif request.node.get_closest_marker('lmstudio'):
            # Force LM Studio - will skip if not available
            force_lmstudio = True
            config = request.getfixturevalue('lmstudio_config')
        else:
            # Default to mock for reliability
            config = request.getfixturevalue('mock_llm_config')
    else:
        config = request.getfixturevalue('mock_llm_config')
    
    # Only test LM Studio connection if explicitly requested
    if force_lmstudio and config["default_provider"] == "lmstudio":
        try:
            # Test connection
            provider = OpenAILLMProvider(
                api_key="not-needed",
                config=config["llm_providers"]["lmstudio"]
            )
            # Try a simple request with longer timeout
            response = await asyncio.wait_for(
                provider.generate("Say 'OK' if you can hear me."),
                timeout=10.0  # Increased timeout
            )
            print(f"\nLM Studio connected successfully: {response[:50]}...\n")
        except asyncio.TimeoutError:
            pytest.skip("LM Studio connection timed out - is the server running?")
        except Exception as e:
            pytest.skip(f"LM Studio not available: {type(e).__name__}: {e}")
            
    return LLMService(config)


@pytest.fixture
async def memory(temp_dir, llm_service) -> AsyncGenerator[Memory, None]:
    """Create a memory system with temporary database and LLM service."""
    memory = Memory(str(temp_dir), llm_service=llm_service)
    yield memory


@pytest.fixture
async def memory_no_llm(temp_dir) -> AsyncGenerator[Memory, None]:
    """Create a memory system without LLM service for basic tests."""
    memory = Memory(str(temp_dir))
    yield memory


@pytest.fixture
async def feedback_processor(memory) -> FeedbackProcessor:
    """Create a feedback processor."""
    return FeedbackProcessor(memory)


@pytest.fixture
async def sample_interactions(memory):
    """Create sample interactions for testing."""
    # Store complete interactions in the new format
    interactions = [
        {
            "interaction_id": "test_1",
            "session_id": "test_session",
            "timestamp": datetime.now().isoformat(),
            "user_query": "Fix the login bug",
            "actions_taken": [
                {"action": "read_file", "file": "auth/login.py", "status": "success"},
                {"action": "edit_file", "file": "auth/login.py", "status": "success"},
                {"action": "run_tests", "target": "auth", "status": "success"}
            ],
            "success": True,
            "feedback_type": None
        },
        {
            "interaction_id": "test_2",
            "session_id": "test_session",
            "timestamp": datetime.now().isoformat(),
            "user_query": "Update the user model",
            "actions_taken": [
                {"action": "read_file", "file": "auth/models.py", "status": "success"},
                {"action": "edit_file", "file": "auth/models.py", "status": "success"},
                {"action": "run_tests", "target": "auth", "status": "failed"}
            ],
            "success": False,
            "feedback_type": "CORRECTION"
        }
    ]
    
    # Store using the new complete interaction format
    for interaction in interactions:
        await memory.store.add_complete_interaction(interaction)
    
    return interactions


@pytest.fixture
async def sample_patterns(memory):
    """Create sample patterns for testing."""
    patterns = [
        {
            "name": "TDD Workflow",
            "description": "Test-driven development pattern",
            "pattern_type": "action_sequence",
            "sequence": ["write_test", "run_test", "implement", "run_test"],
            "confidence": 0.85,
            "frequency": 10
        },
        {
            "name": "Debug Pattern",
            "description": "Common debugging workflow",
            "pattern_type": "action_sequence", 
            "sequence": ["read_file", "edit_file", "run_tests"],
            "confidence": 0.75,
            "frequency": 8
        },
        {
            "name": "File Pair: auth",
            "description": "Files often accessed together",
            "pattern_type": "file_access",
            "sequence": ["auth/login.py", "auth/models.py"],
            "confidence": 0.9,
            "frequency": 15
        }
    ]
    
    for pattern in patterns:
        await memory.store.add_pattern(pattern)
    
    return patterns


@pytest.fixture
async def sample_preferences(memory):
    """Create sample preferences for testing."""
    preferences = [
        ("indent_size", "2", "style"),
        ("test_framework", "pytest", "testing"),
        ("async_io", "always use async/await", "style"),
        ("naming", "snake_case for functions", "style"),
    ]
    
    for key, value, category in preferences:
        await memory.store.add_preference({
            "key": key,
            "value": value,
            "category": category,
            "confidence": 1.0,
            "source": "test",
            "timestamp": "2024-01-01"
        })
    
    return preferences


@pytest.fixture
def diverse_interactions():
    """Diverse interactions for testing world model discovery."""
    return [
        {
            "interaction_id": "cook_1",
            "session_id": "cooking_session",
            "timestamp": datetime.now().isoformat(),
            "user_query": "Help me create a chocolate cake recipe",
            "actions_taken": [
                {"action": "create", "target": "chocolate_cake.md", "status": "success"},
                {"action": "write", "content": "ingredients and steps", "status": "success"}
            ],
            "success": True
        },
        {
            "interaction_id": "quantum_1",
            "session_id": "physics_session",
            "timestamp": datetime.now().isoformat(),
            "user_query": "Simulate a quantum harmonic oscillator",
            "actions_taken": [
                {"action": "calculate", "target": "energy_levels", "status": "success"},
                {"action": "plot", "target": "wavefunctions", "status": "success"}
            ],
            "success": True
        },
        {
            "interaction_id": "music_1",
            "session_id": "music_session",
            "timestamp": datetime.now().isoformat(),
            "user_query": "Compose a melody in C major",
            "actions_taken": [
                {"action": "create", "file": "melody.mid", "status": "success"},
                {"action": "add", "element": "chord_progression", "status": "success"}
            ],
            "success": True
        }
    ]


# Test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "lmstudio: mark test to use LM Studio")
    config.addinivalue_line("markers", "mock: mark test to use mock LLM")
    config.addinivalue_line("markers", "openai: mark test to use OpenAI")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark test as integration test")