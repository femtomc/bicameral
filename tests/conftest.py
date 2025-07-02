"""Pytest configuration and fixtures."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator
from datetime import datetime

from bicamrl.core.memory import Memory
from bicamrl.core.pattern_detector import PatternDetector
from bicamrl.core.feedback_processor import FeedbackProcessor

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
async def memory(temp_dir) -> AsyncGenerator[Memory, None]:
    """Create a memory system with temporary database."""
    memory = Memory(str(temp_dir))
    yield memory

@pytest.fixture
async def pattern_detector(memory) -> PatternDetector:
    """Create a pattern detector."""
    return PatternDetector(memory)

@pytest.fixture
async def feedback_processor(memory) -> FeedbackProcessor:
    """Create a feedback processor."""
    return FeedbackProcessor(memory)

@pytest.fixture
async def sample_interactions(memory):
    """Create sample interactions for testing."""
    # Store some raw interactions in the old format for legacy tests
    interactions = [
        ("read_file", "auth/login.py", {"lines": 150}),
        ("edit_file", "auth/login.py", {"changes": 5}),
        ("run_tests", None, {"passed": True}),
        ("read_file", "auth/models.py", {"lines": 200}),
        ("edit_file", "auth/models.py", {"changes": 3}),
        ("run_tests", None, {"passed": False}),
        ("read_file", "auth/login.py", {"lines": 155}),
        ("run_tests", None, {"passed": True}),
    ]
    
    # Store in the old format directly via SQLite
    for action, file_path, details in interactions:
        await memory.store.add_interaction({
            'action': action,
            'file_path': file_path,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'session_id': 'test_session'
        })
    
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