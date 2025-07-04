"""Test count-based memory consolidation."""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from bicamrl.core.memory import Memory
from bicamrl.core.memory_consolidator import MemoryConsolidator


@pytest.fixture
async def memory_with_consolidator():
    """Create memory with consolidator configured for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(tmpdir)
        # Configure consolidator with lower thresholds for testing
        memory.consolidator.active_to_working_threshold = 5
        memory.consolidator.working_to_episodic_threshold = 3
        memory.consolidator.episodic_to_semantic_threshold = 3
        memory.consolidator.min_frequency_for_semantic = 3
        yield memory


@pytest.mark.asyncio
async def test_count_based_active_to_working(memory_with_consolidator):
    """Test consolidation from active to working memory based on counts."""
    memory = memory_with_consolidator

    # Add interactions - need at least 5 for consolidation
    session_id = "test_session_1"
    for i in range(6):
        await memory.store.add_interaction({
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'action': 'edit_file',
            'file_path': f'test_file_{i % 2}.py',
            'details': {'line': i * 10}
        })

    # Run consolidation
    stats = await memory.consolidate_memories()

    # Should have consolidated 5 interactions into 1 working memory
    assert stats['active_to_working'] >= 1

    # Check that working memory was created
    all_patterns = await memory.get_all_patterns()
    working_memories = [p for p in all_patterns if p.get('pattern_type') == 'consolidated_working']
    assert len(working_memories) >= 1

    # Check working memory has proper metadata
    wm = working_memories[0]
    metadata = wm.get('metadata', {})
    assert metadata.get('type') == 'work_session'
    assert 'files_touched' in metadata
    assert 'action_summary' in metadata


@pytest.mark.asyncio
async def test_working_to_episodic_consolidation(memory_with_consolidator):
    """Test consolidation from working to episodic memory."""
    memory = memory_with_consolidator

    # First create enough interactions to generate multiple working memories
    # Use overlapping files so they'll group together
    for session_num in range(4):
        session_id = f"test_session_{session_num}"
        for i in range(5):
            # Use same file for sessions 0-2, different for session 3
            file_path = 'shared_module.py' if session_num < 3 else 'other_module.py'
            await memory.store.add_interaction({
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'action': 'edit_file',
                'file_path': file_path,
                'details': {'change': f'feature_{session_num}_{i}'}
            })

    # First consolidation creates working memories
    stats1 = await memory.consolidate_memories()
    assert stats1['active_to_working'] >= 3  # Should create at least 3 working memories

    # Second consolidation should create episodic memory
    stats2 = await memory.consolidate_memories()
    assert stats2['working_to_episodic'] >= 1

    # Check episodic memory was created
    all_patterns = await memory.get_all_patterns()
    episodic_memories = [p for p in all_patterns if p.get('pattern_type') == 'consolidated_episodic']
    assert len(episodic_memories) >= 1


@pytest.mark.asyncio
async def test_pattern_based_semantic_extraction(memory_with_consolidator):
    """Test semantic extraction from high-frequency patterns."""
    memory = memory_with_consolidator

    # Create a recurring pattern
    pattern_sequence = ['read_file', 'edit_file', 'run_tests', 'commit']

    # Add the pattern multiple times to reach semantic threshold
    for i in range(4):
        await memory.store.add_pattern({
            'name': 'TDD Pattern',
            'description': 'Test-driven development workflow',
            'pattern_type': 'workflow',
            'sequence': pattern_sequence,
            'frequency': 4,
            'confidence': 0.9
        })

    # Run consolidation
    stats = await memory.consolidate_memories()

    # Should extract semantic knowledge
    assert stats['episodic_to_semantic'] >= 1

    # Check semantic knowledge was created
    all_patterns = await memory.get_all_patterns()
    semantic_patterns = [p for p in all_patterns if p.get('pattern_type') == 'consolidated_semantic']
    assert len(semantic_patterns) >= 1

    # Check it's a semantic pattern
    sp = semantic_patterns[0]
    metadata = sp.get('metadata', {})
    assert metadata.get('type') == 'semantic_pattern'
    assert 'Principle' in sp.get('name', '')


@pytest.mark.asyncio
async def test_no_consolidation_below_threshold(memory_with_consolidator):
    """Test that consolidation doesn't happen below thresholds."""
    memory = memory_with_consolidator

    # Add fewer interactions than threshold
    for i in range(3):  # Below threshold of 5
        await memory.store.add_interaction({
            'timestamp': datetime.now().isoformat(),
            'session_id': 'test_session',
            'action': 'read_file',
            'file_path': 'test.py'
        })

    # Run consolidation
    stats = await memory.consolidate_memories()

    # Should not consolidate anything
    assert stats['active_to_working'] == 0
    assert stats['working_to_episodic'] == 0


@pytest.mark.asyncio
async def test_consolidation_tracks_source_interactions(memory_with_consolidator):
    """Test that consolidated memories track their source interactions."""
    memory = memory_with_consolidator

    # Add interactions (note: basic add_interaction doesn't preserve interaction_id)
    timestamps = []
    for i in range(5):
        timestamp = datetime.now().isoformat()
        interaction = {
            'timestamp': timestamp,
            'session_id': 'test_session',
            'action': 'edit_file',
            'file_path': 'test.py'
        }
        await memory.store.add_interaction(interaction)
        timestamps.append(timestamp)

    # Run consolidation
    stats = await memory.consolidate_memories()
    assert stats['active_to_working'] >= 1

    # Check working memory tracks source interactions
    all_patterns = await memory.get_all_patterns()
    working_memories = [p for p in all_patterns if p.get('pattern_type') == 'consolidated_working']

    wm = working_memories[0]
    source_ids = wm.get('metadata', {}).get('source_interactions', [])

    # Should have tracked 5 source interactions
    assert len(source_ids) == 5

    # Source IDs should be in format timestamp_action
    for sid in source_ids:
        assert '_edit_file' in sid


@pytest.mark.asyncio
async def test_multi_session_consolidation(memory_with_consolidator):
    """Test consolidation handles multiple sessions correctly."""
    memory = memory_with_consolidator

    # Add interactions from multiple sessions
    sessions = ['session_a', 'session_b', 'session_c']

    for session in sessions:
        for i in range(5):
            await memory.store.add_interaction({
                'timestamp': datetime.now().isoformat(),
                'session_id': session,
                'action': 'edit_file',
                'file_path': f'{session}_file.py',
                'details': {'change': f'edit_{i}'}
            })

    # Run consolidation
    stats = await memory.consolidate_memories()

    # Should create one working memory per session
    assert stats['active_to_working'] == 3

    # Check each session got its own working memory
    all_patterns = await memory.get_all_patterns()
    working_memories = [p for p in all_patterns if p.get('pattern_type') == 'consolidated_working']
    assert len(working_memories) == 3

    # Each should have different session_id in metadata
    session_ids = set()
    for wm in working_memories:
        sid = wm.get('metadata', {}).get('session_id')
        session_ids.add(sid)

    assert len(session_ids) == 3


@pytest.mark.asyncio
async def test_file_based_grouping_for_episodes(memory_with_consolidator):
    """Test that working memories are grouped by file overlap for episodes."""
    memory = memory_with_consolidator

    # Create working memories with file overlap
    # Group 1: Works on auth.py
    for i in range(5):
        await memory.store.add_interaction({
            'timestamp': datetime.now().isoformat(),
            'session_id': f'auth_session_{i // 5}',
            'action': 'edit_file',
            'file_path': 'auth.py',
            'details': {'feature': 'login'}
        })

    # Group 2: Works on database.py
    for i in range(5):
        await memory.store.add_interaction({
            'timestamp': datetime.now().isoformat(),
            'session_id': f'db_session_{i // 5}',
            'action': 'edit_file',
            'file_path': 'database.py',
            'details': {'feature': 'migration'}
        })

    # Group 3: Works on auth.py again (should group with Group 1)
    for i in range(5):
        await memory.store.add_interaction({
            'timestamp': datetime.now().isoformat(),
            'session_id': f'auth_session_2_{i // 5}',
            'action': 'edit_file',
            'file_path': 'auth.py',
            'details': {'feature': 'permissions'}
        })

    # First consolidation creates working memories
    stats1 = await memory.consolidate_memories()
    assert stats1['active_to_working'] == 3

    # Second consolidation should group by files
    stats2 = await memory.consolidate_memories()
    # With threshold of 3, we might not get episodic yet, but test the grouping logic

    # Check the grouping logic directly
    all_patterns = await memory.get_all_patterns()
    working_memories = [p for p in all_patterns if p.get('pattern_type') == 'consolidated_working']

    groups = memory.consolidator._group_working_memories(working_memories)

    # Should have 2 groups (auth.py related and database.py related)
    assert len(groups) >= 2

    # Check that auth.py memories are grouped together
    auth_group = None
    for group in groups:
        files = set()
        for wm in group:
            files.update(wm.get('metadata', {}).get('files_touched', []))
        if 'auth.py' in files:
            auth_group = group
            break

    assert auth_group is not None
    # Auth group should have memories from both auth sessions
    assert len([wm for wm in auth_group if 'auth.py' in wm.get('metadata', {}).get('files_touched', [])]) >= 2
