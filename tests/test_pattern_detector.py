"""Tests for pattern detection - Updated for current system."""

import pytest
from datetime import datetime, timedelta

from bicamrl.core.pattern_detector import PatternDetector

@pytest.mark.asyncio
async def test_file_pattern_detection(pattern_detector, memory):
    """Test detection of file access patterns."""
    # Create a pattern of files accessed together using old-style interactions
    for _ in range(5):
        await memory.store.add_interaction({
            "action": "read_file",
            "file_path": "config.py",
            "timestamp": datetime.now().isoformat(),
            "session_id": memory.session_id,
            "details": {}
        })
        await memory.store.add_interaction({
            "action": "read_file", 
            "file_path": "settings.py",
            "timestamp": datetime.now().isoformat(),
            "session_id": memory.session_id,
            "details": {}
        })
    
    patterns = await pattern_detector.check_for_patterns()
    
    # Should detect file pair pattern
    file_patterns = [p for p in patterns if p['pattern_type'] == 'file_access']
    assert len(file_patterns) > 0
    
    # Check the pattern details
    pair_pattern = next((p for p in file_patterns if 'config.py' in p['sequence']), None)
    assert pair_pattern is not None
    assert 'settings.py' in pair_pattern['sequence']
    assert pair_pattern['frequency'] >= 3

@pytest.mark.asyncio
async def test_action_sequence_detection(pattern_detector, memory):
    """Test detection of action sequence patterns."""
    # Create a repeated action sequence
    sequence = ["read_file", "edit_file", "run_tests", "commit"]
    
    for _ in range(4):
        for action in sequence:
            await memory.store.add_interaction({
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "session_id": memory.session_id,
                "details": {}
            })
    
    patterns = await pattern_detector.check_for_patterns()
    
    # Should detect action sequences
    action_patterns = [p for p in patterns if p['pattern_type'] == 'action_sequence']
    assert len(action_patterns) > 0
    
    # Look for our specific sequence or subsequences
    found_sequence = False
    for pattern in action_patterns:
        if all(action in pattern['sequence'] for action in ["read_file", "edit_file"]):
            found_sequence = True
            break
    
    assert found_sequence

@pytest.mark.asyncio
async def test_workflow_pattern_detection(pattern_detector, memory):
    """Test detection of workflow patterns."""
    # Create workflow with time-based grouping
    base_time = datetime.now()
    
    # Workflow 1
    workflows = [
        [
            ("read_file", "test.py", base_time),
            ("edit_file", "test.py", base_time + timedelta(seconds=30)),
            ("run_tests", None, base_time + timedelta(seconds=60)),
        ],
        # Gap to separate workflows
        [
            ("read_file", "test.py", base_time + timedelta(minutes=10)),
            ("edit_file", "test.py", base_time + timedelta(minutes=10, seconds=30)),
            ("run_tests", None, base_time + timedelta(minutes=11)),
        ],
        [
            ("read_file", "test.py", base_time + timedelta(minutes=20)),
            ("edit_file", "test.py", base_time + timedelta(minutes=20, seconds=30)),
            ("run_tests", None, base_time + timedelta(minutes=21)),
        ]
    ]
    
    # Log workflows
    for workflow in workflows:
        for action, file_path, timestamp in workflow:
            await memory.store.add_interaction({
                "action": action,
                "file_path": file_path,
                "timestamp": timestamp.isoformat(),
                "session_id": "test",
                "details": {}
            })
    
    patterns = await pattern_detector.check_for_patterns()
    
    # Should detect workflow patterns
    workflow_patterns = [p for p in patterns if p['pattern_type'] == 'workflow']
    assert len(workflow_patterns) > 0

@pytest.mark.asyncio
async def test_pattern_matching(pattern_detector, sample_patterns):
    """Test finding patterns that match a given sequence."""
    # Test exact match
    matches = await pattern_detector.find_matching_patterns(
        ["write_test", "run_test", "implement", "run_test"]
    )
    assert len(matches) > 0
    assert matches[0]['match_type'] == 'exact'
    
    # Test partial match - the Debug Pattern contains ["read_file", "edit_file", "run_tests"]
    # So we need to provide a longer sequence that contains the pattern
    matches = await pattern_detector.find_matching_patterns(
        ["read_file", "edit_file", "run_tests", "commit"]
    )
    assert len(matches) > 0
    
    # Should match the debug pattern as a subsequence
    debug_match = next((m for m in matches if "Debug" in m['pattern']['name']), None)
    assert debug_match is not None
    assert debug_match['match_type'] in ['subsequence', 'fuzzy']

@pytest.mark.asyncio
async def test_pattern_confidence_update(pattern_detector, memory, sample_patterns):
    """Test updating pattern confidence based on feedback."""
    # Get initial pattern
    patterns = await memory.get_all_patterns()
    pattern = patterns[0]
    initial_confidence = pattern['confidence']
    
    # Update confidence
    await pattern_detector.update_pattern_confidence(pattern['id'], 0.1)
    
    # Check updated confidence
    updated_patterns = await memory.get_all_patterns()
    updated_pattern = next(p for p in updated_patterns if p['id'] == pattern['id'])
    
    assert updated_pattern['confidence'] == min(1.0, initial_confidence + 0.1)
    
    # Test confidence bounds
    await pattern_detector.update_pattern_confidence(pattern['id'], 10.0)
    patterns = await memory.get_all_patterns()
    pattern = next(p for p in patterns if p['id'] == pattern['id'])
    assert pattern['confidence'] == 1.0  # Should be capped at 1.0

@pytest.mark.asyncio 
async def test_minimum_frequency_threshold(pattern_detector, memory):
    """Test that patterns below minimum frequency are not detected."""
    # Create pattern that appears only twice (below threshold of 3)
    for _ in range(2):
        await memory.store.add_interaction({
            "action": "rare_action",
            "timestamp": datetime.now().isoformat(),
            "session_id": memory.session_id,
            "details": {}
        })
        await memory.store.add_interaction({
            "action": "another_rare_action",
            "timestamp": datetime.now().isoformat(),
            "session_id": memory.session_id,
            "details": {}
        })
    
    patterns = await pattern_detector.check_for_patterns()
    
    # Should not detect patterns with frequency < 3
    rare_patterns = [p for p in patterns if "rare_action" in str(p)]
    assert len(rare_patterns) == 0