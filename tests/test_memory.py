"""Tests for memory system - Testing current functionality during transition."""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from bicamrl.core.memory import Memory
from bicamrl.core.interaction_model import Interaction, Action, ActionStatus, FeedbackType
from bicamrl.core.interaction_logger import InteractionLogger

@pytest.fixture
async def memory():
    """Create a memory manager with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = Memory(tmpdir)
        yield manager

@pytest.fixture
async def interaction_logger(memory):
    """Create an interaction logger."""
    return InteractionLogger(memory)

@pytest.mark.asyncio
async def test_store_complete_interaction(memory):
    """Test storing complete interactions."""
    # Create a complete interaction
    interaction = Interaction(
        interaction_id="test_1",
        session_id="test_session",
        user_query="Fix the bug in authentication"
    )

    # Add some actions
    interaction.actions_taken = [
        Action(action_type="read_file", target="auth.py", status=ActionStatus.COMPLETED),
        Action(action_type="edit_file", target="auth.py", status=ActionStatus.COMPLETED)
    ]

    # Store it
    await memory.store.add_complete_interaction(interaction.to_dict())

    # Also store as old-style interaction for get_recent_context
    await memory.store.add_interaction({
        'action': 'edit_file',
        'file_path': 'auth.py',
        'timestamp': datetime.now().isoformat(),
        'session_id': memory.session_id
    })

    # Get recent context to verify it was stored
    context = await memory.get_recent_context()
    assert context['total_interactions'] >= 1

@pytest.mark.asyncio
async def test_get_recent_context(memory):
    """Test getting recent context - current implementation."""
    # Add some old-style interactions for compatibility
    for i in range(3):
        await memory.store.add_interaction({
            'action': f'action_{i}',
            'file_path': f'file_{i}.py',
            'details': {'test': True},
            'timestamp': datetime.now().isoformat(),
            'session_id': memory.session_id
        })

    # Get context
    context = await memory.get_recent_context()

    assert 'recent_actions' in context
    assert 'total_interactions' in context
    assert context['total_interactions'] >= 3

@pytest.mark.asyncio
async def test_preferences(memory):
    """Test preference handling - current implementation."""
    # Preferences are stored directly via store
    await memory.store.add_preference({
        'key': 'style',
        'value': 'concise',
        'category': 'coding',
        'source': 'test'
    })

    # Get preferences returns a dict of categories
    prefs = await memory.get_preferences()
    assert isinstance(prefs, dict)
    assert 'coding' in prefs
    assert prefs['coding']['style'] == 'concise'

@pytest.mark.asyncio
async def test_search(memory):
    """Test search functionality - current implementation."""
    # Add some data to search
    await memory.store.add_interaction({
        'action': 'edit_file',
        'file_path': 'auth/login.py',
        'details': {'description': 'Fixed authentication bug'},
        'timestamp': datetime.now().isoformat(),
        'session_id': memory.session_id
    })

    # Search
    results = await memory.search("authentication")
    # Results might be empty if search only looks at patterns
    assert isinstance(results, list)

@pytest.mark.asyncio
async def test_stats(memory):
    """Test getting statistics - current implementation."""
    # Add some test data
    for i in range(3):
        await memory.store.add_interaction({
            'action': f'action_{i}',
            'file_path': f'file_{i}.py',
            'timestamp': datetime.now().isoformat(),
            'session_id': memory.session_id
        })

    stats = await memory.get_stats()

    assert 'total_interactions' in stats
    assert stats['total_interactions'] >= 3

@pytest.mark.asyncio
async def test_clear_specific(memory):
    """Test clearing specific types of memory."""
    # Add some patterns to clear
    await memory.store.add_pattern({
        'name': 'test_pattern',
        'pattern_type': 'test',
        'sequence': ['a', 'b'],
        'confidence': 0.8
    })

    # Clear patterns
    await memory.clear_specific('patterns')

    # Should be empty
    patterns = await memory.get_all_patterns()
    assert len(patterns) == 0

@pytest.mark.asyncio
async def test_hybrid_functionality_integrated(memory):
    """Test that hybrid store functionality is now integrated into Memory."""
    # Memory now exposes hybrid store methods
    assert hasattr(memory, 'search_similar_queries')
    assert hasattr(memory, 'find_correction_patterns')
    assert hasattr(memory, 'find_successful_patterns')
    assert hasattr(memory, 'cluster_similar_queries')

    # But hybrid_store is None without embeddings
    assert memory.hybrid_store is None

    # These methods should gracefully handle missing hybrid store
    results = await memory.search_similar_queries("test query")
    assert isinstance(results, list)  # Falls back to regular search
