"""Test dynamic world model inference using LLMs."""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from bicamrl.core.memory import Memory
from bicamrl.core.dynamic_world_model import (
    DynamicWorldModelInferencer, DynamicWorldState, 
    DynamicEntity, DynamicRelation
)
from bicamrl.core.llm_service import LLMService, LLMResponse
from bicamrl.core.dynamic_memory_consolidator import DynamicMemoryConsolidator


@pytest.fixture
async def memory(tmp_path):
    """Create a memory instance for testing."""
    mem = Memory(tmp_path / "test_memory")
    yield mem


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service for testing."""
    service = Mock(spec=LLMService)
    
    # Mock world model inference
    async def mock_infer_world_model(interaction):
        # Simulate different domains based on query content
        query = interaction.get('user_query', '').lower()
        
        if 'recipe' in query or 'cook' in query:
            domain = 'cooking'
            entities = [
                {"id": "chocolate_cake", "type": "recipe", "properties": {"difficulty": "medium"}},
                {"id": "oven", "type": "equipment", "properties": {"temperature": "350F"}}
            ]
            relations = [
                {"source": "chocolate_cake", "target": "oven", "type": "requires"}
            ]
            goals = [
                {"type": "creation", "description": "Bake a chocolate cake", "confidence": 0.9}
            ]
        elif 'quantum' in query or 'physics' in query:
            domain = 'quantum_physics'
            entities = [
                {"id": "qubit", "type": "quantum_system", "properties": {"state": "superposition"}},
                {"id": "hamiltonian", "type": "operator", "properties": {"hermitian": True}}
            ]
            relations = [
                {"source": "hamiltonian", "target": "qubit", "type": "acts_on"}
            ]
            goals = [
                {"type": "simulation", "description": "Simulate quantum system", "confidence": 0.8}
            ]
        elif 'music' in query or 'compose' in query:
            domain = 'music_composition'
            entities = [
                {"id": "melody", "type": "musical_element", "properties": {"key": "C major"}},
                {"id": "harmony", "type": "musical_element", "properties": {"progression": "I-IV-V"}}
            ]
            relations = [
                {"source": "harmony", "target": "melody", "type": "supports"}
            ]
            goals = [
                {"type": "composition", "description": "Create a musical piece", "confidence": 0.85}
            ]
        else:
            # Default/unknown domain
            domain = 'general_assistance'
            entities = [
                {"id": "task", "type": "objective", "properties": {"status": "in_progress"}}
            ]
            relations = []
            goals = [
                {"type": "assistance", "description": query[:50], "confidence": 0.6}
            ]
            
        return LLMResponse(
            content={
                "domain": domain,
                "confidence": 0.8,
                "entities": entities,
                "relations": relations,
                "goals": goals
            },
            raw_response=json.dumps({"domain": domain}),
            request_id="test_123",
            duration_ms=100,
            provider="mock"
        )
    
    service.infer_world_model = AsyncMock(side_effect=mock_infer_world_model)
    return service


@pytest.fixture
def world_inferencer(mock_llm_service):
    """Create a world model inferencer with mock LLM."""
    return DynamicWorldModelInferencer(mock_llm_service)


@pytest.mark.asyncio
async def test_dynamic_domain_discovery(world_inferencer):
    """Test that the system can discover any domain dynamically."""
    
    # Test 1: Cooking domain (never hardcoded!)
    cooking_interaction = {
        "interaction_id": "cook_1",
        "user_query": "Help me make a chocolate cake recipe",
        "actions_taken": [
            {"action": "create", "target": "recipe.md"},
            {"action": "write", "content": "ingredients list"}
        ],
        "success": True
    }
    
    world_state = await world_inferencer.infer_from_interaction(cooking_interaction)
    
    assert world_state.domain == "cooking"
    assert "recipe" in world_state.discovered_entity_types
    assert "equipment" in world_state.discovered_entity_types
    assert "requires" in world_state.discovered_relation_types
    
    # Test 2: Quantum physics domain
    physics_interaction = {
        "interaction_id": "phys_1",
        "user_query": "Simulate a quantum harmonic oscillator",
        "actions_taken": [
            {"action": "calculate", "target": "eigenvalues"},
            {"action": "plot", "target": "wavefunction"}
        ],
        "success": True
    }
    
    world_state = await world_inferencer.infer_from_interaction(physics_interaction)
    
    # Should update to new domain
    assert world_state.domain == "quantum_physics"
    assert "quantum_system" in world_state.discovered_entity_types
    assert "operator" in world_state.discovered_entity_types
    
    # Test 3: Music composition domain
    music_interaction = {
        "interaction_id": "music_1",
        "user_query": "Help me compose a melody in C major",
        "actions_taken": [
            {"action": "create", "target": "melody.mid"},
            {"action": "add", "content": "chord progression"}
        ],
        "success": True
    }
    
    world_state = await world_inferencer.infer_from_interaction(music_interaction)
    
    assert world_state.domain == "music_composition"
    assert "musical_element" in world_state.discovered_entity_types
    assert "supports" in world_state.discovered_relation_types


@pytest.mark.asyncio
async def test_dynamic_entity_type_discovery(world_inferencer):
    """Test that entity types are discovered dynamically, not from enums."""
    
    # Simulate an interaction in a completely novel domain
    interaction = {
        "interaction_id": "novel_1",
        "user_query": "Help me design a permaculture garden layout",
        "actions_taken": [
            {"action": "create", "target": "garden_plan.svg"},
            {"action": "add", "element": "compost_bin"},
            {"action": "add", "element": "rain_barrel"}
        ],
        "success": True
    }
    
    # Mock a creative LLM response
    world_inferencer.llm_service.infer_world_model.return_value = LLMResponse(
        content={
            "domain": "permaculture_design",
            "confidence": 0.85,
            "entities": [
                {"id": "garden_plan.svg", "type": "design_document", "properties": {"format": "svg"}},
                {"id": "compost_bin", "type": "sustainable_infrastructure", "properties": {"function": "waste_recycling"}},
                {"id": "rain_barrel", "type": "water_management_system", "properties": {"capacity": "50_gallons"}},
                {"id": "soil", "type": "living_system", "properties": {"health": "building"}}
            ],
            "relations": [
                {"source": "compost_bin", "target": "soil", "type": "enriches"},
                {"source": "rain_barrel", "target": "garden_plan.svg", "type": "integrated_into"},
                {"source": "soil", "target": "rain_barrel", "type": "benefits_from"}
            ],
            "goals": [
                {"type": "sustainable_design", "description": "Create self-sustaining garden", "confidence": 0.9}
            ]
        },
        raw_response="{}",
        request_id="test_123",
        duration_ms=150,
        provider="mock"
    )
    
    world_state = await world_inferencer.infer_from_interaction(interaction)
    
    # Check that completely novel entity types were discovered
    assert world_state.domain == "permaculture_design"
    assert "design_document" in world_state.discovered_entity_types
    assert "sustainable_infrastructure" in world_state.discovered_entity_types
    assert "water_management_system" in world_state.discovered_entity_types
    assert "living_system" in world_state.discovered_entity_types
    
    # Check novel relation types
    assert "enriches" in world_state.discovered_relation_types
    assert "integrated_into" in world_state.discovered_relation_types
    assert "benefits_from" in world_state.discovered_relation_types


@pytest.mark.asyncio
async def test_llm_based_memory_consolidation(memory, mock_llm_service):
    """Test that memory consolidation uses LLM for pattern discovery."""
    
    consolidator = DynamicMemoryConsolidator(memory, mock_llm_service)
    
    # Add diverse interactions
    interactions = [
        {
            "interaction_id": f"int_{i}",
            "session_id": "test_session",
            "timestamp": datetime.now().isoformat(),
            "user_query": f"Working on music theory concept {i}",
            "actions_taken": [
                {"action": "analyze", "target": f"chord_{i}"}
            ],
            "success": True,
            "data": json.dumps({"domain_hint": "music"})
        }
        for i in range(15)
    ]
    
    for interaction in interactions:
        await memory.store.add_complete_interaction(**interaction)
    
    # Mock LLM responses for consolidation
    mock_llm_service._execute_request = AsyncMock(
        return_value=LLMResponse(
            content={
                "sessions": [
                    {
                        "indices": list(range(15)),
                        "goal": "Understanding music theory progressions",
                        "confidence": 0.9
                    }
                ]
            },
            raw_response="{}",
            request_id="test_consolidation",
            duration_ms=200,
            provider="mock"
        )
    )
    
    # Run consolidation
    stats = await consolidator.consolidate_memories()
    
    # Should have created world model
    assert stats["world_models_updated"] > 0
    
    # Check that patterns were created
    patterns = await memory.store.get_patterns()
    assert len(patterns) > 0


@pytest.mark.asyncio
async def test_insights_generation(world_inferencer):
    """Test that the system generates insights about discovered patterns."""
    
    # Add multiple interactions to build up a world model
    interactions = [
        {
            "interaction_id": f"garden_{i}",
            "user_query": f"Add {item} to garden plan",
            "actions_taken": [{"action": "add", "target": item}],
            "success": True
        }
        for i, item in enumerate(["tomatoes", "basil", "peppers", "compost", "trellis"])
    ]
    
    # Process interactions
    for interaction in interactions:
        await world_inferencer.infer_from_interaction(interaction)
    
    # Get insights
    insights = world_inferencer.get_insights()
    
    assert insights["total_entities"] > 0
    assert insights["total_relations"] >= 0
    assert len(insights["discovered_entity_types"]) > 0
    assert insights["interaction_count"] == 5
    assert "goal_achievement_rate" in insights
    assert "most_connected_entities" in insights


@pytest.mark.asyncio
async def test_fallback_for_llm_failure(world_inferencer):
    """Test that system gracefully handles LLM failures."""
    
    # Mock LLM failure
    world_inferencer.llm_service.infer_world_model.return_value = LLMResponse(
        content=None,
        raw_response="",
        request_id="failed_123",
        duration_ms=0,
        error="LLM service unavailable"
    )
    
    interaction = {
        "interaction_id": "fallback_1",
        "user_query": "Create a new document",
        "actions_taken": [
            {"action": "create", "file": "document.txt"}
        ],
        "success": True
    }
    
    # Should still work with basic inference
    world_state = await world_inferencer.infer_from_interaction(interaction)
    
    assert len(world_state.entities) > 0
    assert "document.txt" in world_state.entities
    assert len(world_state.inferred_goals) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])