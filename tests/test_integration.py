"""Integration tests for the complete Bicamrl system."""

import asyncio
import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

from bicamrl.core.feedback_processor import FeedbackProcessor
from bicamrl.core.memory import Memory
from bicamrl.core.pattern_detector import PatternDetector
from bicamrl.server import (
    cleanup_server,
    complete_interaction,
    consolidate_memories,
    detect_pattern,
    feedback_processor,
    get_memory_insights,
    get_memory_stats,
    get_relevant_context,
    initialize_server,
    log_action,
    log_ai_interpretation,
    memory,
    pattern_detector,
    record_feedback,
    search_memory,
    sleep_layer,
    start_interaction,
)
from bicamrl.sleep.config_validator import SleepConfigValidator


class TestBicameralIntegration:
    """Integration tests for the full Bicamrl system."""

    @pytest.fixture
    async def test_system(self):
        """Set up a complete test system."""
        # Create temporary directory
        test_dir = tempfile.mkdtemp()
        test_path = Path(test_dir)

        # Create test configuration
        config = {
            "sleep": {
                "enabled": True,
                "batch_size": 5,
                "analysis_interval": 1,  # Fast for testing
                "min_confidence": 0.6,
                "llm_providers": {"mock": {"type": "mock"}},
                "roles": {
                    "analyzer": "mock",
                    "generator": "mock",
                    "enhancer": "mock",
                    "optimizer": "mock",
                },
            }
        }

        # Save config
        config_path = test_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Initialize components
        import bicamrl.server as server

        server.config = config
        server.memory = Memory(str(test_path / "memory"))
        server.pattern_detector = PatternDetector(server.memory)
        server.feedback_processor = FeedbackProcessor(server.memory)

        # Initialize Sleep Layer if needed
        if config.get("sleep", {}).get("enabled"):
            from bicamrl.sleep.llm_providers import create_llm_providers
            from bicamrl.sleep.sleep import Sleep

            sleep_config = SleepConfigValidator.validate(config["sleep"])
            llm_providers = create_llm_providers(sleep_config)
            server.sleep_layer = Sleep(server.memory, llm_providers, sleep_config)
            await server.sleep_layer.start()

        yield server

        # Cleanup
        if server.sleep_layer:
            await server.sleep_layer.stop()

        shutil.rmtree(test_dir)

    @pytest.mark.asyncio
    async def test_full_workflow_cycle(self, test_system):
        """Test a complete workflow from interaction to pattern detection to feedback."""
        server = test_system

        # 1. Log a series of interactions representing a common workflow
        workflow_interactions = [
            ("open_file", "main.py", {"purpose": "start_feature"}),
            ("read_file", "main.py", {"lines": 100}),
            ("edit_file", "main.py", {"changes": 5}),
            ("save_file", "main.py", {"success": True}),
            ("run_tests", None, {"passed": True}),
            ("commit", None, {"message": "Add feature"}),
        ]

        # Log the workflow 3 times (minimum for pattern detection)
        for i in range(3):
            for action, file_path, details in workflow_interactions:
                result = await log_interaction(action=action, file_path=file_path, details=details)
                assert "logged" in result.lower()

            # Add time gap between workflows
            await asyncio.sleep(0.1)

        # 2. Check that patterns were detected
        patterns = await server.memory.get_all_patterns()
        assert len(patterns) > 0

        # Look for workflow patterns
        workflow_patterns = [p for p in patterns if p.get("pattern_type") == "workflow"]
        assert len(workflow_patterns) > 0, "Should detect workflow patterns"

        # 3. Test pattern matching
        test_sequence = ["open_file", "read_file", "edit_file"]
        matches = await detect_pattern(action_sequence=test_sequence)
        assert len(matches) > 0, "Should find matching patterns"

        # 4. Test context retrieval
        context = await get_relevant_context(
            task_description="implement new feature", file_context=["main.py"]
        )
        assert "relevant_patterns" in context
        assert "recent_context" in context

        # 5. Record feedback
        feedback_result = await record_feedback(
            feedback_type="prefer", message="always run tests before committing"
        )
        assert "recorded" in feedback_result.lower()

        # 6. Verify feedback affects preferences
        preferences = await server.memory.get_preferences()
        assert len(preferences) > 0

        # 7. Test memory search
        search_results = await search_memory("main.py")
        assert len(search_results) > 0
        assert any("main.py" in str(r) for r in search_results)

        # 8. Test memory stats
        stats = await get_memory_stats()
        assert stats["total_interactions"] > 0
        assert stats["total_patterns"] > 0

    @pytest.mark.asyncio
    async def test_memory_consolidation(self, test_system):
        """Test memory consolidation from active to semantic."""
        server = test_system

        # Create interactions at different time periods
        base_time = datetime.now()

        # Old interactions (should be consolidated)
        old_time = base_time - timedelta(days=8)
        for i in range(10):
            await server.memory.store.add_interaction(
                {
                    "session_id": "old_session",
                    "action": "old_action",
                    "file_path": f"old_file_{i}.py",
                    "details": {},
                    "timestamp": (old_time + timedelta(minutes=i)).isoformat(),
                }
            )

        # Recent interactions (should stay active)
        for i in range(5):
            await log_interaction(action="recent_action", file_path=f"recent_file_{i}.py")

        # Run consolidation
        consolidation_stats = await consolidate_memories()
        assert "statistics" in consolidation_stats

        # Check consolidated memories
        consolidated = await server.memory.get_consolidated_memories()
        assert len(consolidated) > 0

        # Test memory insights
        insights = await get_memory_insights("working with old files")
        assert "consolidated_memories" in insights
        assert "recent_context" in insights

    @pytest.mark.asyncio
    async def test_pattern_evolution(self, test_system):
        """Test how patterns evolve with feedback and new data."""
        server = test_system

        # Create initial pattern
        initial_sequence = ["edit", "test", "fix", "test", "commit"]
        for _ in range(3):
            for action in initial_sequence:
                await log_interaction(action=action)

        # Get initial patterns
        patterns_before = await server.memory.get_all_patterns()
        initial_pattern = next(
            (p for p in patterns_before if "test" in str(p.get("sequence", []))), None
        )
        assert initial_pattern is not None
        initial_confidence = initial_pattern.get("confidence", 0)

        # Add negative feedback
        await record_feedback(feedback_type="correct", message="should run linter before tests")

        # Create improved pattern
        improved_sequence = ["edit", "lint", "test", "fix", "test", "commit"]
        for _ in range(3):
            for action in improved_sequence:
                await log_interaction(action=action)

        # Check pattern confidence changed
        patterns_after = await server.memory.get_all_patterns()

        # New pattern should exist
        new_pattern = next(
            (p for p in patterns_after if "lint" in str(p.get("sequence", []))), None
        )
        assert new_pattern is not None
        assert new_pattern.get("confidence", 0) > 0.7

    @pytest.mark.asyncio
    async def test_concurrent_access(self, test_system):
        """Test system behavior under concurrent access."""
        server = test_system

        async def simulate_user(user_id: int, num_interactions: int):
            """Simulate a user making interactions."""
            for i in range(num_interactions):
                await log_interaction(
                    action=f"action_{i}",
                    file_path=f"user_{user_id}_file_{i}.py",
                    details={"user": user_id},
                )

                if i % 5 == 0:
                    # Occasionally search
                    await search_memory(f"user_{user_id}")

                if i % 10 == 0:
                    # Occasionally check patterns
                    await detect_pattern([f"action_{i - 1}", f"action_{i}"])

        # Simulate 5 concurrent users
        users = []
        for i in range(5):
            users.append(simulate_user(i, 20))

        # Run all users concurrently
        await asyncio.gather(*users)

        # Verify system handled concurrent access
        stats = await get_memory_stats()
        assert stats["total_interactions"] >= 100  # 5 users * 20 interactions

        # Check no data corruption
        all_interactions = await server.memory.store.get_recent_interactions(200)
        user_counts = {}
        for interaction in all_interactions:
            user_id = interaction.get("details", {}).get("user")
            if user_id is not None:
                user_counts[user_id] = user_counts.get(user_id, 0) + 1

        # Each user should have their interactions
        assert len(user_counts) == 5
        assert all(count == 20 for count in user_counts.values())

    @pytest.mark.asyncio
    async def test_error_recovery(self, test_system):
        """Test system behavior when errors occur."""
        server = test_system

        # Test invalid feedback type
        with pytest.raises(ValueError):
            await record_feedback(feedback_type="invalid_type", message="test")

        # Test pattern detection with empty sequence
        result = await detect_pattern(action_sequence=[])
        assert isinstance(result, list)
        assert len(result) == 0

        # Test search with very long query
        long_query = "a" * 1000
        results = await search_memory(long_query)
        assert isinstance(results, list)  # Should handle gracefully

        # Test context retrieval with missing parameters
        context = await get_relevant_context(task_description="", file_context=None)
        assert isinstance(context, dict)

        # System should still be functional after errors
        result = await log_interaction(action="test_after_errors", file_path="test.py")
        assert "logged" in result.lower()

    @pytest.mark.asyncio
    async def test_preference_learning(self, test_system):
        """Test how the system learns and applies preferences."""
        server = test_system

        # Record multiple preferences
        preferences_to_learn = [
            ("prefer", "use 4 spaces for indentation"),
            ("prefer", "always add type hints"),
            ("prefer", "use pytest for testing"),
            ("correct", "don't use global variables"),
        ]

        for feedback_type, message in preferences_to_learn:
            await record_feedback(feedback_type=feedback_type, message=message)

        # Get learned preferences
        preferences = await server.memory.get_preferences()
        assert len(preferences) > 0

        # Test that preferences affect context
        context = await get_relevant_context(
            task_description="write unit tests", file_context=["test_example.py"]
        )

        # Should include testing preferences
        assert "applicable_preferences" in context
        prefs = context["applicable_preferences"]

        # Verify preferences are categorized
        assert any("pytest" in str(v) for category in prefs.values() for v in category.values())

    @pytest.mark.asyncio
    async def test_cross_session_persistence(self, test_system):
        """Test that patterns persist across sessions."""
        server = test_system

        # Create patterns in "session 1"
        session1_pattern = ["login", "navigate", "edit", "save", "logout"]
        for _ in range(3):
            for action in session1_pattern:
                await log_interaction(action=action)

        # Get patterns from session 1
        patterns1 = await server.memory.get_all_patterns()
        assert len(patterns1) > 0

        # Simulate new session by changing session_id
        original_session = server.memory.session_id
        server.memory.session_id = datetime.now().isoformat()

        # Patterns should still be available
        patterns2 = await server.memory.get_all_patterns()
        assert len(patterns2) == len(patterns1)

        # New session can find patterns from old session
        matches = await detect_pattern(action_sequence=["login", "navigate"])
        assert len(matches) > 0

        # Restore session ID
        server.memory.session_id = original_session
