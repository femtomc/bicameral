"""Test the unified role proposer that uses world models."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

from bicamrl.sleep.role_proposer import RoleProposer, generate_role_name, CULTURAL_NAMES
from bicamrl.sleep.roles import CommandRole, ContextTrigger, TriggerType
from bicamrl.core.memory import Memory
from bicamrl.core.world_model import WorldModelInferencer
from bicamrl.core.llm_service import LLMService, LLMResponse


class TestRoleProposer:
    """Test the role proposer functionality."""

    def test_generate_role_name(self):
        """Test culturally diverse name generation."""
        # Generate 100 names and check they're from the cultural lists
        names = [generate_role_name() for _ in range(100)]

        # Check all names are from the cultural lists
        all_valid_names = []
        for culture_names in CULTURAL_NAMES.values():
            all_valid_names.extend(culture_names)

        for name in names:
            assert name in all_valid_names

        # Check we get diversity (at least 10 different names in 100 tries)
        assert len(set(names)) >= 10

    @pytest.mark.asyncio
    async def test_discover_roles_no_world_models(self):
        """Test role discovery when no world models exist."""
        # Mock dependencies
        memory = Mock(spec=Memory)
        memory.store = Mock()
        memory.store.get_world_model_states = AsyncMock(return_value=[])

        world_model = Mock(spec=WorldModelInferencer)
        llm_service = Mock(spec=LLMService)

        proposer = RoleProposer(memory, world_model, llm_service)

        roles = await proposer.discover_roles()

        assert roles == []
        memory.store.get_world_model_states.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_roles_with_world_models(self):
        """Test role discovery with world model data."""
        # Mock world model states
        world_states = [
            {
                "domain": "software_development",
                "entities": [{"id": "test.py", "type": "file"}],
                "relations": [{"source": "user", "target": "test.py", "type": "edits"}],
                "goals": [{"goal": "debugging", "confidence": 0.8}]
            }
        ]

        # Mock memory
        memory = Mock(spec=Memory)
        memory.store = Mock()
        memory.store.get_world_model_states = AsyncMock(return_value=world_states)
        memory.get_recent_interactions = AsyncMock(return_value=[
            {
                "query": "Fix the bug in test.py",
                "actions": ["read_file", "edit_file"],
                "success": True
            }
        ])

        # Mock world model
        world_model = Mock(spec=WorldModelInferencer)

        # Mock LLM service
        llm_service = Mock(spec=LLMService)

        # Mock world insights response
        llm_service.infer_world_model = AsyncMock(return_value=LLMResponse(
            content={
                "domains": [{"name": "software_development", "frequency": 0.8}],
                "recurring_goals": [{"goal": "debugging", "domain": "software_development"}],
                "entity_patterns": [],
                "behavioral_insights": ["User frequently debugs Python code"]
            },
            raw_response="",
            request_id="test",
            duration_ms=100
        ))

        # Mock pattern analysis response
        llm_service.analyze_patterns = AsyncMock(return_value=LLMResponse(
            content={
                "patterns": [{
                    "pattern_id": "debug_pattern_1",
                    "domain": "software_development",
                    "recurring_goals": ["debugging"],
                    "key_entities": ["test.py"],
                    "action_sequences": [["read_file", "edit_file"]],
                    "success_indicators": {"bug_fixed": True},
                    "context_conditions": {"file_type": "python"},
                    "frequency": 0.8,
                    "confidence": 0.9
                }]
            },
            raw_response="",
            request_id="test",
            duration_ms=100
        ))

        # Mock role creation response
        llm_service.enhance_prompt = AsyncMock(return_value=LLMResponse(
            content='''{
                "description": "Python debugging specialist",
                "mindset_prompt": "You are a Python debugging expert. Focus on identifying and fixing bugs efficiently.",
                "context_triggers": [
                    {"condition": "file_pattern", "value": "\\\\.py$", "weight": 0.8}
                ],
                "communication_style": {
                    "tone": "technical",
                    "formality": 0.7,
                    "verbosity": 0.5,
                    "technical_depth": 0.9
                },
                "tool_preferences": {"read_file": 0.9, "edit_file": 0.8},
                "decision_rules": [
                    {"condition": "bug reported", "action": "analyze code systematically", "confidence": 0.9}
                ]
            }''',
            raw_response="",
            request_id="test",
            duration_ms=100
        ))

        proposer = RoleProposer(memory, world_model, llm_service)

        roles = await proposer.discover_roles(min_interactions=1)

        # Verify a role was created
        assert len(roles) == 1
        role = roles[0]

        # Check role has a culturally diverse name
        assert role.name in [name for names in CULTURAL_NAMES.values() for name in names]

        # Check role properties
        assert role.description == "Python debugging specialist"
        assert role.mindset_prompt == "You are a Python debugging expert. Focus on identifying and fixing bugs efficiently."
        assert role.domain == "software_development"
        assert role.discovered_from_pattern == "debug_pattern_1"

        # Check context triggers
        assert len(role.context_triggers) == 1
        assert role.context_triggers[0].trigger_type == "file_pattern"
        assert role.context_triggers[0].pattern == "\\.py$"

        # Check tool preferences
        assert role.tool_preferences["read_file"] == 0.9
        assert role.tool_preferences["edit_file"] == 0.8

    @pytest.mark.asyncio
    async def test_refine_role(self):
        """Test role refinement based on feedback."""
        # Create a test role
        role = CommandRole(
            name="TestRole",
            description="Original description",
            mindset_prompt="Original mindset",
            domain="test_domain"
        )

        # Mock dependencies
        memory = Mock(spec=Memory)
        world_model = Mock(spec=WorldModelInferencer)
        llm_service = Mock(spec=LLMService)

        # Mock refinement response
        llm_service.enhance_prompt = AsyncMock(return_value=LLMResponse(
            content='''{
                "description": "Refined description",
                "mindset_prompt": "Refined mindset prompt with better focus",
                "context_triggers": [
                    {"condition": "task_keyword", "value": "debug,fix,error", "weight": 0.9}
                ],
                "communication_style": {
                    "tone": "professional",
                    "formality": 0.8,
                    "verbosity": 0.3,
                    "technical_depth": 0.8
                }
            }''',
            raw_response="",
            request_id="test",
            duration_ms=100
        ))

        proposer = RoleProposer(memory, world_model, llm_service)

        feedback = {
            "success_rate": 0.85,
            "common_failures": ["missed edge cases"],
            "user_feedback": "needs to be more thorough"
        }

        refined_role = await proposer.refine_role(role, feedback)

        # Check refinements were applied
        assert refined_role.description == "Refined description"
        assert refined_role.mindset_prompt == "Refined mindset prompt with better focus"
        assert len(refined_role.context_triggers) == 1
        assert refined_role.context_triggers[0].trigger_type == "task_keyword"
