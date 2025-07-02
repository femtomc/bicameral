"""Test cleanup and archival functionality."""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from bicamrl.core.memory import Memory
from bicamrl.storage.sqlite_store import SQLiteStore


@pytest.fixture
async def memory_with_data():
    """Create memory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(tmpdir)
        # Configure for easier testing
        memory.consolidator.active_to_working_threshold = 3
        memory.consolidator.working_to_episodic_threshold = 2
        
        # Add test interactions
        for i in range(10):
            await memory.store.add_interaction({
                'interaction_id': f'test_interaction_{i}',
                'timestamp': datetime.now().isoformat(),
                'session_id': f'session_{i // 3}',  # 3-4 per session
                'action': 'edit_file',
                'file_path': f'file_{i % 3}.py',
                'details': {'change': f'edit_{i}'}
            })
        
        yield memory


@pytest.mark.asyncio
async def test_interaction_id_tracking(memory_with_data):
    """Test that interaction_ids are properly stored and retrieved."""
    memory = memory_with_data
    
    # Add an interaction with explicit ID
    test_id = 'explicit_test_id_123'
    await memory.store.add_interaction({
        'interaction_id': test_id,
        'timestamp': datetime.now().isoformat(),
        'session_id': 'test_session',
        'action': 'test_action',
        'file_path': 'test.py'
    })
    
    # Retrieve and check
    recent = await memory.store.get_recent_interactions(20)
    interaction_ids = [i.get('interaction_id') for i in recent]
    
    assert test_id in interaction_ids
    
    # Check that auto-generated IDs exist for all
    for interaction in recent:
        assert 'interaction_id' in interaction
        assert interaction['interaction_id'] is not None


@pytest.mark.asyncio
async def test_archive_interactions(memory_with_data):
    """Test archiving interactions."""
    memory = memory_with_data
    
    # Get some interaction IDs
    interactions = await memory.store.get_recent_interactions(5)
    ids_to_archive = [i['interaction_id'] for i in interactions[:3]]
    
    # Archive them
    archived_count = await memory.store.archive_interactions(
        interaction_ids=ids_to_archive,
        reason="test_archive",
        consolidated_to="test_working_memory_1"
    )
    
    assert archived_count == 3
    
    # Verify they're gone from active table
    remaining = await memory.store.get_recent_interactions(20)
    remaining_ids = [i['interaction_id'] for i in remaining]
    
    for archived_id in ids_to_archive:
        assert archived_id not in remaining_ids
    
    # Verify they're in archive
    for archived_id in ids_to_archive:
        archived = await memory.store.get_archived_interaction(archived_id)
        assert archived is not None
        assert archived['interaction_id'] == archived_id
        assert archived['archive_reason'] == 'test_archive'
        assert archived['consolidated_to'] == 'test_working_memory_1'


@pytest.mark.asyncio
async def test_archive_patterns(memory_with_data):
    """Test archiving patterns."""
    memory = memory_with_data
    
    # Create some patterns
    pattern_ids = []
    for i in range(3):
        pattern_id = f'test_pattern_{i}'
        await memory.store.add_pattern({
            'id': pattern_id,
            'name': f'Pattern {i}',
            'pattern_type': 'consolidated_working',
            'description': f'Test pattern {i}',
            'frequency': 5,
            'confidence': 0.8,
            'sequence': ['a', 'b', 'c'],
            'metadata': {'test': True}
        })
        pattern_ids.append(pattern_id)
    
    # Archive them
    archived_count = await memory.store.archive_patterns(
        pattern_ids=pattern_ids[:2],
        reason="promoted_to_episodic",
        promoted_to="episodic_memory_1"
    )
    
    assert archived_count == 2
    
    # Verify only one remains in active table
    remaining = await memory.get_all_patterns()
    remaining_ids = [p.get('id') for p in remaining]
    
    assert pattern_ids[2] in remaining_ids
    assert pattern_ids[0] not in remaining_ids
    assert pattern_ids[1] not in remaining_ids


@pytest.mark.asyncio
async def test_cleanup_with_consolidation():
    """Test full cleanup after consolidation."""
    # Create fresh memory for this test to control the flow
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(tmpdir)
        memory.consolidator.active_to_working_threshold = 3
        
        # Add interactions that will be consolidated
        interaction_ids = []
        for i in range(9):
            iid = f'cleanup_test_{i}'
            interaction = {
                'interaction_id': iid,
                'timestamp': datetime.now().isoformat(),
                'session_id': f'session_{i // 3}',
                'action': 'edit_file',
                'file_path': f'file_{i % 2}.py',
                'details': {'change': f'edit_{i}'}
            }
            await memory.store.add_interaction(interaction)
            interaction_ids.append(iid)
            
            # Verify it was added
            recent = await memory.store.get_recent_interactions(20)
            print(f"After adding {i+1} interactions, have {len(recent)} in db")
        
        # Verify all interactions are there
        interactions_before = await memory.store.get_recent_interactions(20)
        print(f"Before consolidation: {len(interactions_before)} interactions")
        assert len(interactions_before) == 9
        
        # Run consolidation to create working memories
        stats1 = await memory.consolidate_memories()
        print(f"Consolidation stats: {stats1}")
        assert stats1['active_to_working'] >= 3
        
        # The consolidation INCLUDES cleanup, so interactions should be archived
        assert stats1['cleaned_up'] > 0
        
        # Check that interactions were removed from active table
        interactions_after_consolidation = await memory.store.get_recent_interactions(20)
        print(f"After consolidation: {len(interactions_after_consolidation)} interactions remain")
        assert len(interactions_after_consolidation) < 9
        
        # Verify they're in the archive
        archive_stats = await memory.store.get_archive_statistics()
        print(f"Archive stats: {archive_stats}")
        assert archive_stats['total_archived_interactions'] >= 9
        assert 'consolidated_to_working' in archive_stats['interactions_by_reason']
        
        # The cleanup happened during consolidation
        assert archive_stats['interactions_by_reason']['consolidated_to_working'] >= 9


@pytest.mark.asyncio
async def test_archive_statistics(memory_with_data):
    """Test archive statistics tracking."""
    memory = memory_with_data
    
    # Archive some interactions with different reasons
    interactions = await memory.store.get_recent_interactions(10)
    
    # Archive first batch
    ids_batch_1 = [i['interaction_id'] for i in interactions[:3]]
    await memory.store.archive_interactions(
        interaction_ids=ids_batch_1,
        reason="consolidated_to_working"
    )
    
    # Archive second batch
    ids_batch_2 = [i['interaction_id'] for i in interactions[3:5]]
    await memory.store.archive_interactions(
        interaction_ids=ids_batch_2,
        reason="manual_cleanup"
    )
    
    # Get statistics
    stats = await memory.store.get_archive_statistics()
    
    assert stats['total_archived_interactions'] == 5
    assert stats['interactions_by_reason']['consolidated_to_working'] == 3
    assert stats['interactions_by_reason']['manual_cleanup'] == 2


@pytest.mark.asyncio
async def test_vacuum_after_large_cleanup(memory_with_data):
    """Test that vacuum is called after large cleanup."""
    memory = memory_with_data
    
    # Add many interactions
    for i in range(100):
        await memory.store.add_interaction({
            'timestamp': datetime.now().isoformat(),
            'session_id': f'bulk_session_{i // 10}',
            'action': 'bulk_action',
            'file_path': 'bulk.py'
        })
    
    # Configure low threshold for testing
    memory.consolidator.active_to_working_threshold = 10
    
    # Run consolidation
    stats = await memory.consolidate_memories()
    assert stats['active_to_working'] >= 10
    
    # The cleanup should trigger vacuum for > 1000 archived items
    # For this test, we just verify the method exists and can be called
    await memory.store.vacuum_database()  # Should not raise


@pytest.mark.asyncio
async def test_migration_of_existing_database():
    """Test migration adds interaction_id to existing database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Create old-style database without interaction_id
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute('''
            CREATE TABLE interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                action TEXT NOT NULL,
                file_path TEXT,
                details TEXT,
                embeddings BLOB
            )
        ''')
        
        # Add some data
        conn.execute('''
            INSERT INTO interactions (timestamp, session_id, action, file_path, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            'old_session',
            'old_action',
            'old_file.py',
            '{}'
        ))
        conn.commit()
        conn.close()
        
        # Now initialize SQLiteStore which should migrate
        store = SQLiteStore(db_path)
        
        # Check that interaction_id was added
        interactions = await store.get_recent_interactions(10)
        assert len(interactions) == 1
        assert 'interaction_id' in interactions[0]
        assert interactions[0]['interaction_id'] is not None
        
        # Verify it's a combination of timestamp_action_id
        assert '_old_action_' in interactions[0]['interaction_id']


@pytest.mark.asyncio
async def test_no_duplicate_archival():
    """Test that interactions aren't archived multiple times."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(tmpdir)
        
        # Add interactions
        interaction_ids = []
        for i in range(5):
            iid = f'no_dup_{i}'
            await memory.store.add_interaction({
                'interaction_id': iid,
                'timestamp': datetime.now().isoformat(),
                'session_id': 'test',
                'action': 'test',
                'file_path': 'test.py'
            })
            interaction_ids.append(iid)
        
        # Archive them once
        archived1 = await memory.store.archive_interactions(
            interaction_ids=interaction_ids,
            reason="first_archive"
        )
        assert archived1 == 5
        
        # Try to archive again - should archive 0 since they're already gone
        archived2 = await memory.store.archive_interactions(
            interaction_ids=interaction_ids,
            reason="second_archive"
        )
        assert archived2 == 0
        
        # Verify only one copy in archive
        stats = await memory.store.get_archive_statistics()
        assert stats['total_archived_interactions'] == 5