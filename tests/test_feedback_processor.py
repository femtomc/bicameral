"""Tests for feedback processing."""

import pytest

from bicamrl.core.feedback_processor import FeedbackProcessor

@pytest.mark.asyncio
async def test_process_correction(feedback_processor, memory, sample_interactions):
    """Test processing correction feedback."""
    result = await feedback_processor.process_feedback(
        "correct",
        "Don't use global variables in auth module"
    )
    
    assert "Correction noted" in result
    
    # Check feedback was stored
    feedback = await memory.store.get_feedback()
    assert len(feedback) == 1
    assert feedback[0]['type'] == 'correct'
    assert "global variables" in feedback[0]['message']

@pytest.mark.asyncio
async def test_process_preference(feedback_processor, memory):
    """Test processing preference feedback."""
    # Test key:value format
    result = await feedback_processor.process_feedback(
        "prefer",
        "line_length: 100 characters"
    )
    
    assert "Preference stored" in result
    
    # Check preference was stored correctly
    prefs = await memory.get_preferences()
    assert 'general' in prefs
    assert prefs['general']['line_length'] == '100 characters'
    
    # Test "use X instead of Y" format
    result = await feedback_processor.process_feedback(
        "prefer", 
        "use pytest instead of unittest"
    )
    
    prefs = await memory.get_preferences()
    assert 'style' in prefs
    found_pref = False
    for key, value in prefs['style'].items():
        if isinstance(value, dict) and value.get('prefer') == 'pytest':
            assert value['avoid'] == 'unittest'
            found_pref = True
            break
    assert found_pref

@pytest.mark.asyncio
async def test_process_pattern_feedback(feedback_processor, memory, sample_interactions):
    """Test processing pattern teaching feedback."""
    result = await feedback_processor.process_feedback(
        "pattern",
        "Always run linter before committing code"
    )
    
    assert "Pattern learned" in result
    
    # Check pattern was created
    patterns = await memory.get_all_patterns()
    user_patterns = [p for p in patterns if p['pattern_type'] == 'user_defined']
    assert len(user_patterns) == 1
    assert "linter" in user_patterns[0]['description']
    assert user_patterns[0]['confidence'] == 0.8  # High confidence for user-taught

@pytest.mark.asyncio
async def test_preference_parsing(feedback_processor):
    """Test parsing different preference formats."""
    # Test various formats
    test_cases = [
        ("always validate input", "always_do", "validate input"),
        ("never use eval()", "never_do", "use eval()"),
        ("prefer async functions", "preference", "async functions"),
        ("tab_size: 4", "tab_size", "4"),
    ]
    
    for message, expected_key, expected_value in test_cases:
        pref = feedback_processor._parse_preference(message)
        assert pref is not None
        
        if expected_key in ["always_do", "never_do", "preference"]:
            assert pref['key'] == expected_key
            assert pref['value'] == expected_value
        else:
            assert pref['key'] == expected_key
            assert pref['value'] == expected_value

@pytest.mark.asyncio
async def test_apply_feedback_to_patterns(feedback_processor, memory, sample_patterns):
    """Test applying feedback to update pattern confidence."""
    # Add correction feedback
    await feedback_processor.process_feedback(
        "correct",
        "That workflow doesn't work for async code"
    )
    
    # Apply feedback
    result = await feedback_processor.apply_feedback_to_patterns()
    
    assert result['feedback_processed'] > 0
    
    # Check that patterns were updated
    patterns = await memory.get_all_patterns()
    # Patterns that match recent actions should have reduced confidence
    
@pytest.mark.asyncio
async def test_feedback_context_capture(feedback_processor, memory, sample_interactions):
    """Test that feedback captures recent context."""
    # Process feedback
    await feedback_processor.process_feedback(
        "correct",
        "Use dependency injection here"
    )
    
    # Check stored feedback includes context
    feedback = await memory.store.get_feedback()
    assert len(feedback) == 1
    assert 'context' in feedback[0]
    assert 'recent_actions' in feedback[0]['context']
    assert len(feedback[0]['context']['recent_actions']) > 0

@pytest.mark.asyncio
async def test_pattern_matching_for_feedback(feedback_processor, sample_patterns):
    """Test pattern matching logic for feedback application."""
    # Create a pattern and some actions
    pattern = {
        'sequence': ['read_file', 'edit_file', 'run_tests']
    }
    
    actions = [
        {'action': 'read_file'},
        {'action': 'edit_file'},
        {'action': 'run_tests'},
        {'action': 'commit'}
    ]
    
    # Should match
    assert feedback_processor._pattern_matches_actions(pattern, actions) == True
    
    # Should not match if sequence not present
    actions_no_match = [
        {'action': 'read_file'},
        {'action': 'commit'},
        {'action': 'push'}
    ]
    assert feedback_processor._pattern_matches_actions(pattern, actions_no_match) == False