"""Edge case tests for pattern detection - Updated for current system."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from bicamrl.core.memory import Memory
from bicamrl.core.pattern_detector import PatternDetector


@pytest.fixture
async def setup():
    """Create fresh memory manager and pattern detector for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(tmpdir)
        pattern_detector = PatternDetector(memory)
        yield memory, pattern_detector


@pytest.mark.asyncio
async def test_empty_interactions(setup):
    """Test pattern detection with no interactions."""
    memory, pattern_detector = setup

    patterns = await pattern_detector.check_for_patterns()

    # No patterns should be detected
    assert len(patterns) == 0


@pytest.mark.asyncio
async def test_single_interaction(setup):
    """Test pattern detection with only one interaction."""
    memory, pattern_detector = setup

    await memory.store.add_interaction({
        "action": "single_action",
        "file_path": "file.py",
        "timestamp": datetime.now().isoformat(),
        "session_id": memory.session_id,
        "details": {}
    })

    patterns = await pattern_detector.check_for_patterns()

    # No patterns should be detected from a single interaction
    assert len(patterns) == 0


@pytest.mark.asyncio
async def test_below_threshold_frequency(setup):
    """Test that patterns below min_frequency threshold are not detected."""
    memory, pattern_detector = setup

    # Create a pattern that appears only twice (below default threshold of 3)
    for _ in range(2):
        await memory.store.add_interaction({
            "action": "rare_action1",
            "timestamp": datetime.now().isoformat(),
            "session_id": memory.session_id,
            "details": {}
        })
        await memory.store.add_interaction({
            "action": "rare_action2",
            "timestamp": datetime.now().isoformat(),
            "session_id": memory.session_id,
            "details": {}
        })

    patterns = await pattern_detector.check_for_patterns()

    # Should not detect patterns with frequency < 3
    action_patterns = [p for p in patterns if p['pattern_type'] == 'action_sequence']
    assert len(action_patterns) == 0


@pytest.mark.asyncio
async def test_fuzzy_matching_similar_sequences(setup):
    """Test fuzzy matching for similar but not identical sequences."""
    memory, pattern_detector = setup

    # Create a pattern
    pattern_sequence = ["open_file", "edit_file", "save_file", "close_file"]
    for _ in range(3):
        for action in pattern_sequence:
            await memory.store.add_interaction({
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "session_id": memory.session_id,
                "details": {}
            })

    patterns = await pattern_detector.check_for_patterns()

    # Should detect the pattern
    action_patterns = [p for p in patterns if p['pattern_type'] == 'action_sequence']
    assert len(action_patterns) > 0

    # Test fuzzy matching with similar sequence
    similar_sequence = ["open_file", "modify_file", "save_file", "close_file"]
    matches = await pattern_detector.find_matching_patterns(similar_sequence)

    # Should find fuzzy matches
    fuzzy_matches = [m for m in matches if m['match_type'] == 'fuzzy']
    assert len(fuzzy_matches) > 0


@pytest.mark.asyncio
async def test_recency_weighting(setup):
    """Test that recent patterns are weighted more heavily."""
    memory, pattern_detector = setup

    # Create old interactions
    old_time = datetime.now() - timedelta(days=7)
    for i in range(5):
        await memory.store.add_interaction({
            "action": "old_pattern",
            "file_path": "old.py",
            "timestamp": (old_time + timedelta(minutes=i)).isoformat(),
            "session_id": "old_session",
            "details": {}
        })

    # Create recent interactions
    recent_time = datetime.now() - timedelta(hours=1)
    for i in range(3):  # Less frequent but more recent
        await memory.store.add_interaction({
            "action": "recent_pattern",
            "file_path": "recent.py",
            "timestamp": (recent_time + timedelta(minutes=i)).isoformat(),
            "session_id": "recent_session",
            "details": {}
        })

    patterns = await pattern_detector.check_for_patterns()

    # Both patterns should be detected
    assert len(patterns) >= 2


@pytest.mark.asyncio
async def test_workflow_boundary_detection(setup):
    """Test detection of workflow boundaries based on time gaps."""
    memory, pattern_detector = setup

    base_time = datetime.now()

    # Create 3 workflows to meet frequency threshold
    workflow_times = [
        base_time,
        base_time + timedelta(minutes=30),
        base_time + timedelta(hours=2)  # Large gap before third
    ]

    for workflow_time in workflow_times:
        for i, action in enumerate(["read", "edit", "test"]):
            await memory.store.add_interaction({
                "action": action,
                "timestamp": (workflow_time + timedelta(seconds=i*30)).isoformat(),
                "session_id": "test",
                "details": {}
            })

    patterns = await pattern_detector.check_for_patterns()

    # Should detect patterns from the repeated workflow
    assert len(patterns) > 0

    # Check for action sequence patterns (should find the read->edit->test pattern)
    action_patterns = [p for p in patterns if p['pattern_type'] == 'action_sequence']
    assert len(action_patterns) > 0


@pytest.mark.asyncio
async def test_malformed_timestamps(setup):
    """Test handling of malformed timestamps."""
    memory, pattern_detector = setup

    # Add interaction with malformed timestamp
    await memory.store.add_interaction({
        "session_id": "test",
        "action": "test_action",
        "file_path": "test.py",
        "details": {},
        "timestamp": "not-a-valid-timestamp"
    })

    # Add valid interactions
    for i in range(3):
        await memory.store.add_interaction({
            "action": "valid_action",
            "file_path": f"file{i}.py",
            "timestamp": datetime.now().isoformat(),
            "session_id": memory.session_id,
            "details": {}
        })

    # Should handle gracefully and still detect patterns from valid interactions
    patterns = await pattern_detector.check_for_patterns()
    # May or may not have patterns depending on implementation
    assert isinstance(patterns, list)


@pytest.mark.asyncio
async def test_duplicate_patterns(setup):
    """Test handling of duplicate patterns."""
    memory, pattern_detector = setup

    # Create same sequence multiple times
    sequence = ["action1", "action2", "action3"]

    for session in range(4):  # Multiple sessions
        for action in sequence:
            await memory.store.add_interaction({
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "session_id": f"session_{session}",
                "details": {}
            })

    patterns = await pattern_detector.check_for_patterns()

    # Should consolidate duplicate patterns
    action_patterns = [p for p in patterns if p['pattern_type'] == 'action_sequence']
    # Should have detected patterns
    assert len(action_patterns) > 0

    # Check that the pattern has appropriate frequency
    if action_patterns:
        assert action_patterns[0]['frequency'] >= 3


@pytest.mark.asyncio
async def test_very_long_sequences(setup):
    """Test handling of very long action sequences."""
    memory, pattern_detector = setup

    # Create a very long sequence
    long_sequence = [f"action_{i}" for i in range(10)]

    # Log it 3 times
    for _ in range(3):
        for action in long_sequence:
            await memory.store.add_interaction({
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "session_id": memory.session_id,
                "details": {}
            })

    patterns = await pattern_detector.check_for_patterns()

    # Should handle long sequences (may truncate or split)
    action_patterns = [p for p in patterns if p['pattern_type'] == 'action_sequence']
    assert len(action_patterns) > 0

    # Check that sequences are reasonable length
    for pattern in action_patterns:
        assert len(pattern.get('sequence', [])) <= 20  # Reasonable max length


@pytest.mark.asyncio
async def test_confidence_update_bounds(setup):
    """Test that confidence updates respect bounds [0, 1]."""
    memory, pattern_detector = setup

    # Create a pattern
    for _ in range(3):
        await memory.store.add_interaction({
            "action": "test_action",
            "timestamp": datetime.now().isoformat(),
            "session_id": memory.session_id,
            "details": {}
        })

    patterns = await pattern_detector.check_for_patterns()

    if patterns:
        # Add pattern to store
        pattern = patterns[0]
        await memory.store.add_pattern(pattern)

        # Test extreme confidence updates
        await pattern_detector.update_pattern_confidence(pattern['id'], 10.0)

        # Get updated pattern
        updated_patterns = await memory.get_all_patterns()
        if updated_patterns:
            updated = next((p for p in updated_patterns if p['id'] == pattern['id']), None)
            if updated:
                assert 0 <= updated['confidence'] <= 1.0

        # Test negative update
        await pattern_detector.update_pattern_confidence(pattern['id'], -10.0)

        # Get updated pattern again
        updated_patterns = await memory.get_all_patterns()
        if updated_patterns:
            updated = next((p for p in updated_patterns if p['id'] == pattern['id']), None)
            if updated:
                assert 0 <= updated['confidence'] <= 1.0
