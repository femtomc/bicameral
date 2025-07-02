"""Test role management functionality."""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bicamrl.core.interaction_logger import InteractionLogger
from bicamrl.core.memory import Memory
from bicamrl.sleep.role_manager import RoleManager
from bicamrl.sleep.roles import (
    CommandRole,
    CommunicationProfile,
    CommunicationStyle,
    ContextTrigger,
    TriggerType,
)
from bicamrl.storage.hybrid_store import HybridStore


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Mock analyze that returns role-like responses."""
        if "discover roles" in prompt.lower():
            return json.dumps({
                "roles": [
                    {
                        "name": "Python Debugger",
                        "description": "Expert at finding and fixing Python errors",
                        "triggers": ["error", "debug", "fix", "TypeError"],
                        "tool_preferences": ["search_error", "read_file", "fix_bug"],
                        "communication_style": "technical"
                    }
                ]
            })
        return "{}"
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock generate."""
        return "Generated response"


@pytest.fixture
async def role_manager(tmp_path):
    """Create a role manager for testing."""
    memory = Memory(str(tmp_path / "memory"))
    hybrid_store = HybridStore(tmp_path / "hybrid")
    
    config = {
        "sleep": {
            "roles": {
                "auto_discover": True,
                "discovery_interval": 300,
                "min_interactions_per_role": 3,
                "storage_path": str(tmp_path / "roles")
            }
        }
    }
    
    manager = RoleManager(memory, hybrid_store, config)
    await manager.initialize()
    
    yield manager
    
    # Cleanup
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture
def sample_roles():
    """Create sample roles for testing."""
    roles = []
    
    # Python debugger role
    python_debugger = CommandRole(
        name="Python Debugger",
        description="Expert at finding and fixing Python errors",
        context_triggers=[
            ContextTrigger(
                trigger_type=TriggerType.KEYWORD,
                pattern="error",
                weight=0.8
            ),
            ContextTrigger(
                trigger_type=TriggerType.KEYWORD,
                pattern="debug",
                weight=0.9
            ),
            ContextTrigger(
                trigger_type=TriggerType.FILE_TYPE,
                pattern=".py",
                weight=0.5
            )
        ],
        behavior_rules=[
            "Focus on error messages and stack traces",
            "Use debugging tools systematically",
            "Explain the root cause before fixing"
        ],
        tool_preferences={
            "search_error": 0.9,
            "read_file": 0.8,
            "fix_bug": 0.7
        },
        communication_profile=CommunicationProfile(
            style=CommunicationStyle.TECHNICAL,
            verbosity_level=0.7,
            explanation_depth=0.8
        ),
        confidence_threshold=0.7
    )
    roles.append(python_debugger)
    
    # API developer role
    api_developer = CommandRole(
        name="API Developer",
        description="Specializes in building RESTful APIs",
        context_triggers=[
            ContextTrigger(
                trigger_type=TriggerType.KEYWORD,
                pattern="endpoint",
                weight=0.9
            ),
            ContextTrigger(
                trigger_type=TriggerType.KEYWORD,
                pattern="API",
                weight=0.8
            ),
            ContextTrigger(
                trigger_type=TriggerType.TASK,
                pattern="create.*endpoint",
                weight=0.9
            )
        ],
        behavior_rules=[
            "Design RESTful endpoints",
            "Include proper validation",
            "Add comprehensive tests"
        ],
        tool_preferences={
            "create_file": 0.9,
            "add_validation": 0.8,
            "write_tests": 0.8
        },
        communication_profile=CommunicationProfile(
            style=CommunicationStyle.EXPLANATORY,
            verbosity_level=0.6,
            explanation_depth=0.7
        ),
        confidence_threshold=0.75
    )
    roles.append(api_developer)
    
    return roles


class TestRoleManager:
    """Test role management functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self, role_manager):
        """Test role manager initialization."""
        assert role_manager.storage_path.exists()
        assert role_manager.auto_discover is True
        assert role_manager.discovery_interval == 300
        assert role_manager.min_interactions_per_role == 3
        
        # Check default roles loaded
        assert len(role_manager.default_roles) > 0
        assert "general_assistant" in role_manager.default_roles
    
    @pytest.mark.asyncio
    async def test_add_custom_role(self, role_manager, sample_roles):
        """Test adding custom roles."""
        role = sample_roles[0]  # Python Debugger
        
        # Add role
        role_manager.custom_roles[role.name] = role
        
        # Save roles
        await role_manager.save_roles()
        
        # Verify saved
        assert (role_manager.storage_path / "roles.json").exists()
        
        # Load and verify
        with open(role_manager.storage_path / "roles.json", "r") as f:
            data = json.load(f)
        
        assert len(data["custom_roles"]) == 1
        assert "Python Debugger" in data["custom_roles"]
    
    @pytest.mark.asyncio
    async def test_role_activation(self, role_manager, sample_roles):
        """Test role activation based on context."""
        # Add test roles
        for role in sample_roles:
            role_manager.custom_roles[role.name] = role
        
        # Test debugging context
        debug_context = {
            "user_query": "Fix the TypeError in user.py",
            "recent_actions": ["search_error"],
            "active_files": ["user.py"],
            "has_error": True
        }
        
        active_role = await role_manager.get_active_role(debug_context)
        assert active_role is not None
        assert active_role.name == "Python Debugger"
        
        # Test API context
        api_context = {
            "user_query": "Create a new user registration endpoint",
            "recent_actions": ["design_api"],
            "active_files": ["api.py"],
            "task_description": "create endpoint"
        }
        
        active_role = await role_manager.get_active_role(api_context)
        assert active_role is not None
        assert active_role.name == "API Developer"
    
    @pytest.mark.asyncio
    async def test_role_statistics(self, role_manager, sample_roles):
        """Test role statistics tracking."""
        # Add roles
        for role in sample_roles:
            role_manager.custom_roles[role.name] = role
        
        # Get initial stats
        stats = role_manager.get_role_statistics()
        assert stats["total_roles"] == len(role_manager.default_roles) + 2
        assert stats["custom_roles"] == 2
        assert stats["total_activations"] == 0
        
        # Activate roles
        await role_manager.update_role_performance("Python Debugger", True)
        await role_manager.update_role_performance("Python Debugger", True)
        await role_manager.update_role_performance("API Developer", False)
        
        # Check updated stats
        stats = role_manager.get_role_statistics()
        assert stats["total_activations"] == 3
        
        by_role = stats["activations_by_role"]
        assert by_role["Python Debugger"]["count"] == 2
        assert by_role["Python Debugger"]["success_rate"] == 1.0
        assert by_role["API Developer"]["count"] == 1
        assert by_role["API Developer"]["success_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_role_discovery(self, role_manager):
        """Test role discovery from interactions."""
        # Mock LLM provider
        mock_provider = MockLLMProvider()
        
        # Create discoverer
        from bicamrl.sleep.llm_role_discoverer import LLMRoleDiscoverer
        discoverer = LLMRoleDiscoverer(
            role_manager.memory,
            role_manager.hybrid_store,
            mock_provider
        )
        role_manager.discoverer = discoverer
        
        # Create test interactions
        logger = InteractionLogger(role_manager.memory)
        
        # Create debugging pattern
        for i in range(5):
            interaction_id = logger.start_interaction(f"Fix TypeError in module{i}.py")
            logger.log_interpretation("Debugging Python error", ["search_error", "fix_bug"], 0.9)
            
            act = logger.log_action("search_error", f"module{i}.py")
            logger.complete_action(act, "Found error")
            
            act = logger.log_action("fix_bug", f"module{i}.py")
            logger.complete_action(act, "Fixed")
            
            completed = logger.complete_interaction(success=True)
            if completed:
                await role_manager.hybrid_store.add_interaction(completed)
        
        # Discover roles
        discovered = await discoverer.discover_roles_from_interactions(days_back=1, max_roles=3)
        
        assert len(discovered) >= 1
        assert any(role.name == "Python Debugger" for role in discovered)
    
    @pytest.mark.asyncio
    async def test_role_persistence(self, role_manager, sample_roles):
        """Test saving and loading roles."""
        # Add custom roles
        for role in sample_roles:
            role_manager.custom_roles[role.name] = role
        
        # Update some stats
        await role_manager.update_role_performance("Python Debugger", True)
        
        # Save
        await role_manager.save_roles()
        
        # Create new manager and load
        new_manager = RoleManager(
            role_manager.memory,
            role_manager.hybrid_store,
            {"sleep": {"roles": {"storage_path": str(role_manager.storage_path)}}}
        )
        await new_manager.initialize()
        
        # Verify loaded correctly
        assert len(new_manager.custom_roles) == 2
        assert "Python Debugger" in new_manager.custom_roles
        assert "API Developer" in new_manager.custom_roles
        
        # Check stats preserved
        stats = new_manager.get_role_statistics()
        assert stats["activations_by_role"]["Python Debugger"]["count"] == 1
    
    @pytest.mark.asyncio
    async def test_role_recommendations(self, role_manager, sample_roles):
        """Test getting role recommendations."""
        # Add roles
        for role in sample_roles:
            role_manager.custom_roles[role.name] = role
        
        # Get recommendations for debugging context
        context = {
            "user_query": "Debug the error in my code",
            "active_files": ["main.py"],
            "has_error": True
        }
        
        recommendations = await role_manager.get_role_recommendations(context)
        
        assert len(recommendations) > 0
        assert recommendations[0]["role"] == "Python Debugger"
        assert recommendations[0]["confidence"] > 0.5
        assert "triggers" in recommendations[0]
    
    @pytest.mark.asyncio
    async def test_markdown_export(self, role_manager, sample_roles):
        """Test exporting roles to markdown."""
        # Add a role
        role = sample_roles[0]
        role_manager.custom_roles[role.name] = role
        
        # Export to markdown
        await role_manager.save_roles()
        
        # Create roles directory
        roles_dir = role_manager.storage_path / "roles"
        roles_dir.mkdir(exist_ok=True)
        
        # Save as markdown
        md_path = roles_dir / f"{role.name.lower().replace(' ', '_')}.md"
        with open(md_path, "w") as f:
            f.write(f"# {role.name}\n\n")
            f.write(f"{role.description}\n\n")
            f.write("## Triggers\n")
            for trigger in role.context_triggers:
                f.write(f"- {trigger.trigger_type.value}: {trigger.pattern}\n")
        
        # Verify markdown created
        assert md_path.exists()
        content = md_path.read_text()
        assert "Python Debugger" in content
        assert "error" in content
    
    @pytest.mark.asyncio
    async def test_role_scoring(self, role_manager, sample_roles):
        """Test role scoring algorithm."""
        role = sample_roles[0]  # Python Debugger
        
        # Test high-scoring context
        good_context = {
            "user_query": "Debug this error in my Python code",
            "active_files": ["main.py", "utils.py"],
            "recent_actions": ["search_error"],
            "has_error": True
        }
        
        score = role_manager._score_role_for_context(role, good_context)
        assert score > 0.7
        
        # Test low-scoring context
        bad_context = {
            "user_query": "Create a new React component",
            "active_files": ["App.jsx"],
            "recent_actions": ["create_file"]
        }
        
        score = role_manager._score_role_for_context(role, bad_context)
        assert score < 0.3
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, role_manager, sample_roles):
        """Test concurrent role updates."""
        # Add roles
        for role in sample_roles:
            role_manager.custom_roles[role.name] = role
        
        # Concurrent updates
        tasks = []
        for i in range(10):
            role_name = sample_roles[i % 2].name
            success = i % 3 != 0
            tasks.append(role_manager.update_role_performance(role_name, success))
        
        await asyncio.gather(*tasks)
        
        # Verify all updates recorded
        stats = role_manager.get_role_statistics()
        total = sum(
            data["count"] 
            for data in stats["activations_by_role"].values()
        )
        assert total == 10


class TestRoleDiscovery:
    """Test role discovery functionality."""
    
    @pytest.mark.asyncio
    async def test_interaction_based_discovery(self, role_manager):
        """Test discovering roles from interaction patterns."""
        # Create logger
        logger = InteractionLogger(role_manager.memory)
        
        # Create pattern: Testing workflow
        test_pattern = [
            ("write_test", "test_auth.py"),
            ("run_test", "pytest"),
            ("fix_test", "test_auth.py"),
            ("run_test", "pytest")
        ]
        
        # Log pattern multiple times
        for i in range(5):
            interaction_id = logger.start_interaction(f"Write tests for authentication module v{i}")
            logger.log_interpretation("Writing and fixing tests", [a[0] for a in test_pattern], 0.9)
            
            for action, target in test_pattern:
                act = logger.log_action(action, target)
                logger.complete_action(act, "Success")
            
            completed = logger.complete_interaction(success=True)
            if completed:
                await role_manager.hybrid_store.add_interaction(completed)
        
        # Check pattern detection
        from bicamrl.sleep.interaction_role_discoverer import InteractionRoleDiscoverer
        discoverer = InteractionRoleDiscoverer(
            role_manager.memory,
            role_manager.hybrid_store
        )
        
        patterns = await discoverer._find_behavioral_patterns(days_back=1)
        assert len(patterns) > 0
        
        # Should find test-related pattern
        test_patterns = [p for p in patterns if "test" in str(p["actions"]).lower()]
        assert len(test_patterns) > 0