"""Test world model persistence functionality."""

import pytest
import asyncio
from datetime import datetime

from bicamrl.core.world_model import WorldModelInferencer, WorldState, Entity, Relation
from bicamrl.storage.sqlite_store import SQLiteStore


@pytest.mark.asyncio
async def test_world_model_persistence(temp_dir, llm_service):
    """Test storing and retrieving world models."""
    store = SQLiteStore(temp_dir / "test.db")
    inferencer = WorldModelInferencer(llm_service, store=store)
    
    # Create an interaction to infer from
    interaction = {
        "interaction_id": "test_1",
        "session_id": "test_session",
        "user_query": "Help me create a chocolate cake recipe",
        "actions_taken": [
            {"action": "create", "target": "chocolate_cake.md"},
            {"action": "write", "content": "ingredients and steps"}
        ],
        "success": True
    }
    
    # Infer world model (this should persist it)
    world_state = await inferencer.infer_from_interaction(interaction)
    
    # Check that it has an ID
    assert world_state.id is not None
    world_id = world_state.id
    
    # Retrieve from storage
    stored_world = await store.get_world_model_state(world_id)
    assert stored_world is not None
    assert stored_world['id'] == world_id
    assert stored_world['session_id'] == "test_session"
    assert stored_world['interaction_count'] == 1


@pytest.mark.asyncio
async def test_world_model_manual_persistence(temp_dir):
    """Test manual world model persistence without inferencer."""
    store = SQLiteStore(temp_dir / "test.db")
    
    # Create a world state manually
    world_state = WorldState(
        id="manual_test_1",
        session_id="manual_session",
        domain="cooking",
        domain_confidence=0.9,
        created_at=datetime.now(),
        interaction_count=5
    )
    
    # Add some entities
    world_state.add_entity("recipe.md", "document", category="recipe")
    world_state.add_entity("user", "person", role="chef")
    
    # Add a relation
    world_state.add_relation("user", "recipe.md", "creates")
    
    # Add a goal
    world_state.inferred_goals.append({
        "type": "recipe_creation",
        "description": "Create a new recipe",
        "confidence": 0.8
    })
    
    # Persist it
    world_id = await store.add_world_model_state(world_state.to_dict())
    
    # Retrieve and verify
    stored = await store.get_world_model_state(world_id)
    assert stored is not None
    assert stored['domain'] == "cooking"
    assert stored['domain_confidence'] == 0.9
    assert len(stored['entities']) == 2
    assert len(stored['relations']) == 1
    assert len(stored['goals']) == 1
    assert 'document' in stored['discovered_entity_types']
    assert 'person' in stored['discovered_entity_types']


@pytest.mark.asyncio
async def test_world_model_queries(temp_dir):
    """Test querying world models by various criteria."""
    store = SQLiteStore(temp_dir / "test.db")
    
    # Add multiple world models
    for i in range(3):
        world_state = {
            'id': f'world_{i}',
            'session_id': 'session_1' if i < 2 else 'session_2',
            'domain': 'cooking' if i == 0 else 'coding',
            'domain_confidence': 0.8,
            'entities': {},
            'relations': [],
            'goals': [],
            'metrics': {},
            'discovered_entity_types': [],
            'discovered_relation_types': [],
            'constraints': {},
            'interaction_count': i + 1
        }
        await store.add_world_model_state(world_state)
    
    # Test get active world models
    active_worlds = await store.get_active_world_models()
    assert len(active_worlds) == 3
    
    # Test filter by session
    session_worlds = await store.get_active_world_models(session_id='session_1')
    assert len(session_worlds) == 2
    
    # Test get by domain
    cooking_worlds = await store.get_world_models_by_domain('cooking')
    assert len(cooking_worlds) == 1
    assert cooking_worlds[0]['id'] == 'world_0'
    
    coding_worlds = await store.get_world_models_by_domain('coding')
    assert len(coding_worlds) == 2


@pytest.mark.asyncio
async def test_world_model_snapshots(temp_dir):
    """Test world model snapshot functionality."""
    store = SQLiteStore(temp_dir / "test.db")
    
    # Create and store a world model
    world_state = {
        'id': 'snapshot_test',
        'session_id': 'test_session',
        'domain': 'music',
        'domain_confidence': 0.7,
        'entities': {},
        'relations': [],
        'goals': [],
        'metrics': {'interaction_count': 1},
        'discovered_entity_types': ['instrument', 'note'],  # Use list instead of set
        'discovered_relation_types': ['plays'],  # Use list instead of set
        'constraints': {}
    }
    await store.add_world_model_state(world_state)
    
    # Take a snapshot
    await store.add_world_model_snapshot(
        'snapshot_test',
        world_state,
        'Initial creation'
    )
    
    # Update the world model
    world_state['domain_confidence'] = 0.9
    world_state['metrics']['interaction_count'] = 2
    await store.add_world_model_state(world_state)
    
    # Take another snapshot
    await store.add_world_model_snapshot(
        'snapshot_test',
        world_state,
        'After confidence update'
    )
    
    # Verify snapshots were stored (would need to add a retrieval method)
    # For now, just verify no errors occurred


@pytest.mark.asyncio
async def test_world_model_deactivation(temp_dir):
    """Test deactivating world models."""
    store = SQLiteStore(temp_dir / "test.db")
    
    # Create an active world model
    world_state = {
        'id': 'deactivation_test',
        'session_id': 'test_session',
        'domain': 'physics',
        'entities': {},
        'relations': [],
        'goals': [],
        'metrics': {},
        'discovered_entity_types': set(),
        'discovered_relation_types': set(),
        'constraints': {},
        'is_active': True
    }
    await store.add_world_model_state(world_state)
    
    # Verify it's active
    active = await store.get_active_world_models()
    assert any(w['id'] == 'deactivation_test' for w in active)
    
    # Deactivate it
    await store.deactivate_world_model('deactivation_test')
    
    # Verify it's no longer active
    active = await store.get_active_world_models()
    assert not any(w['id'] == 'deactivation_test' for w in active)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])