"""Tests for storage layer."""

import pytest
import json
from datetime import datetime

from bicamrl.storage.sqlite_store import SQLiteStore

@pytest.fixture
async def store(temp_dir):
    """Create a SQLite store."""
    return SQLiteStore(temp_dir / "test.db")

@pytest.mark.asyncio
async def test_add_and_get_interactions(store):
    """Test storing and retrieving interactions."""
    # Add interactions
    interaction1 = {
        "timestamp": datetime.now().isoformat(),
        "session_id": "test-session",
        "action": "read_file",
        "file_path": "test.py",
        "details": {"lines": 100}
    }

    interaction2 = {
        "timestamp": datetime.now().isoformat(),
        "session_id": "test-session",
        "action": "edit_file",
        "file_path": "test.py",
        "details": {"changes": 5}
    }

    await store.add_interaction(interaction1)
    await store.add_interaction(interaction2)

    # Retrieve interactions
    interactions = await store.get_recent_interactions(10)

    assert len(interactions) == 2
    assert interactions[0]['action'] == 'edit_file'  # Most recent first
    assert interactions[1]['action'] == 'read_file'
    assert interactions[0]['details']['changes'] == 5

@pytest.mark.asyncio
async def test_add_and_get_patterns(store):
    """Test storing and retrieving patterns."""
    pattern = {
        "name": "Test Pattern",
        "description": "A test pattern",
        "pattern_type": "test",
        "sequence": ["action1", "action2"],
        "frequency": 5,
        "confidence": 0.8
    }

    await store.add_pattern(pattern)

    # Retrieve patterns
    patterns = await store.get_patterns()

    assert len(patterns) == 1
    assert patterns[0]['name'] == "Test Pattern"
    assert patterns[0]['sequence'] == ["action1", "action2"]
    assert patterns[0]['confidence'] == 0.8

    # Test filtering by type
    test_patterns = await store.get_patterns("test")
    assert len(test_patterns) == 1

    other_patterns = await store.get_patterns("other")
    assert len(other_patterns) == 0

@pytest.mark.asyncio
async def test_update_pattern_confidence(store):
    """Test updating pattern confidence."""
    # Add a pattern
    pattern = {
        "id": "test-pattern-1",
        "name": "Test Pattern",
        "pattern_type": "test",
        "sequence": []
    }
    await store.add_pattern(pattern)

    # Update confidence
    await store.update_pattern_confidence("test-pattern-1", 0.95)

    # Check update
    patterns = await store.get_patterns()
    assert patterns[0]['confidence'] == 0.95
    assert patterns[0]['last_seen'] is not None

@pytest.mark.asyncio
async def test_add_and_get_feedback(store):
    """Test storing and retrieving feedback."""
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "type": "correct",
        "message": "Don't use global variables",
        "context": {"recent_files": ["test.py"]}
    }

    await store.add_feedback(feedback)

    # Retrieve feedback
    feedback_items = await store.get_feedback()

    assert len(feedback_items) == 1
    assert feedback_items[0]['type'] == 'correct'
    assert feedback_items[0]['message'] == "Don't use global variables"
    assert feedback_items[0]['context']['recent_files'] == ["test.py"]
    assert feedback_items[0]['applied'] == 0

@pytest.mark.asyncio
async def test_mark_feedback_applied(store):
    """Test marking feedback as applied."""
    # Add feedback
    await store.add_feedback({
        "timestamp": datetime.now().isoformat(),
        "type": "prefer",
        "message": "Use async/await"
    })

    feedback = await store.get_feedback()
    feedback_id = feedback[0]['id']

    # Mark as applied
    await store.mark_feedback_applied(feedback_id)

    # Check it was marked
    updated_feedback = await store.get_feedback()
    assert updated_feedback[0]['applied'] == 1

@pytest.mark.asyncio
async def test_add_and_get_preferences(store):
    """Test storing and retrieving preferences."""
    # Add preferences
    pref1 = {
        "key": "indent_size",
        "value": "4",
        "category": "style",
        "confidence": 1.0,
        "source": "explicit"
    }

    pref2 = {
        "key": "test_style",
        "value": {"framework": "pytest", "style": "BDD"},
        "category": "testing"
    }

    await store.add_preference(pref1)
    await store.add_preference(pref2)

    # Retrieve preferences
    prefs = await store.get_preferences()

    assert len(prefs) == 2

    # Check simple value
    indent_pref = next(p for p in prefs if p['key'] == 'indent_size')
    # String "4" gets parsed as JSON and becomes integer 4
    assert indent_pref['value'] == 4

    # Check JSON value
    test_pref = next(p for p in prefs if p['key'] == 'test_style')
    assert test_pref['value']['framework'] == 'pytest'

@pytest.mark.asyncio
async def test_clear_functions(store):
    """Test clearing specific data types."""
    # Add some data
    await store.add_pattern({"name": "Pattern", "pattern_type": "test", "sequence": []})
    await store.add_preference({"key": "test", "value": "value"})
    await store.add_feedback({"timestamp": "2024-01-01", "type": "test", "message": "test"})

    # Clear patterns
    await store.clear_patterns()
    patterns = await store.get_patterns()
    assert len(patterns) == 0

    # Preferences should still exist
    prefs = await store.get_preferences()
    assert len(prefs) == 1

    # Clear all
    await store.clear_all()
    prefs = await store.get_preferences()
    feedback = await store.get_feedback()
    assert len(prefs) == 0
    assert len(feedback) == 0

@pytest.mark.asyncio
async def test_json_serialization(store):
    """Test JSON serialization of complex data."""
    # Add interaction with complex details
    await store.add_interaction({
        "timestamp": datetime.now().isoformat(),
        "action": "test",
        "details": {
            "nested": {
                "data": [1, 2, 3],
                "flag": True
            }
        }
    })

    interactions = await store.get_recent_interactions()
    assert interactions[0]['details']['nested']['data'] == [1, 2, 3]
    assert interactions[0]['details']['nested']['flag'] == True

@pytest.mark.asyncio
async def test_pattern_replacement(store):
    """Test that patterns can be replaced by ID."""
    pattern_id = "unique-pattern-id"

    # Add initial pattern
    await store.add_pattern({
        "id": pattern_id,
        "name": "Original Pattern",
        "pattern_type": "test",
        "sequence": ["a", "b"],
        "confidence": 0.5
    })

    # Replace with updated pattern
    await store.add_pattern({
        "id": pattern_id,
        "name": "Updated Pattern",
        "pattern_type": "test",
        "sequence": ["c", "d"],
        "confidence": 0.8
    })

    # Should only have one pattern with updated values
    patterns = await store.get_patterns()
    assert len(patterns) == 1
    assert patterns[0]['name'] == "Updated Pattern"
    assert patterns[0]['sequence'] == ["c", "d"]
    assert patterns[0]['confidence'] == 0.8
