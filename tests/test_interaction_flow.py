"""Tests for the new interaction-based flow."""

import pytest
from datetime import datetime

from bicamrl.core.interaction_logger import InteractionLogger
from bicamrl.core.interaction_model import Interaction, Action, ActionStatus, FeedbackType
from bicamrl.core.interaction_pattern_detector import InteractionPatternDetector
from bicamrl.storage.hybrid_store import HybridStore


@pytest.fixture
async def interaction_logger(memory):
    """Create an interaction logger."""
    return InteractionLogger(memory)


@pytest.fixture
async def hybrid_store(tmp_path):
    """Create a hybrid store for testing."""
    return HybridStore(tmp_path / "test_storage")


@pytest.fixture
async def interaction_pattern_detector(memory):
    """Create an interaction pattern detector."""
    return InteractionPatternDetector(memory)


@pytest.mark.asyncio
async def test_full_interaction_cycle(interaction_logger, hybrid_store, interaction_pattern_detector):
    """Test a complete interaction cycle from query to feedback."""
    # 1. Start interaction with user query
    interaction_id = interaction_logger.start_interaction(
        user_query="Fix the bug in the user authentication",
        context={"current_file": "main.py"}
    )
    
    assert interaction_id is not None
    assert interaction_logger.current_interaction is not None
    assert interaction_logger.current_interaction.user_query == "Fix the bug in the user authentication"
    
    # 2. Log AI interpretation
    interaction_logger.log_interpretation(
        interpretation="User wants me to fix a bug in the authentication system",
        planned_actions=["search_files", "read_file", "edit_file"],
        confidence=0.8,
        role="debugger"
    )
    
    assert interaction_logger.current_interaction.ai_interpretation is not None
    assert len(interaction_logger.current_interaction.planned_actions) == 3
    
    # 3. Log actions taken
    action1 = interaction_logger.log_action("search_files", target="auth", details={"pattern": "*.py"})
    interaction_logger.complete_action(action1, result="Found auth/token.py")
    
    action2 = interaction_logger.log_action("read_file", target="auth/token.py")
    interaction_logger.complete_action(action2, result="Found bug on line 42")
    
    action3 = interaction_logger.log_action("edit_file", target="auth/token.py", 
                                           details={"line": 42, "change": "Fixed null check"})
    interaction_logger.complete_action(action3, result="File updated")
    
    assert len(interaction_logger.current_interaction.actions_taken) == 3
    
    # 4. Complete interaction with user feedback
    completed = interaction_logger.complete_interaction(
        feedback="Thanks, that fixed it!",
        feedback_type=FeedbackType.APPROVAL,
        success=True
    )
    
    assert completed is not None
    assert completed.success is True
    assert completed.feedback_type == FeedbackType.APPROVAL
    assert completed.execution_time is not None
    
    # 5. Store in hybrid store
    await hybrid_store.add_interaction(completed)
    
    # 6. Verify we can search for similar queries
    # Note: With random embeddings, we need to search for exact or very similar text
    similar = await hybrid_store.search_similar_queries("Fix the bug in the user authentication", k=3)
    assert len(similar) > 0
    # With random embeddings, we can't rely on similarity ordering, just check it's found
    found_texts = [s[2]["text"] for s in similar]
    assert "Fix the bug in the user authentication" in found_texts
    
    # 7. Detect patterns (pattern detector needs min_frequency=2, so one interaction won't create patterns)
    patterns = await interaction_pattern_detector.detect_patterns([completed])
    # With only one interaction, no patterns will be detected (min_frequency=2)
    assert len(patterns) == 0
    
    # But if we add another similar interaction, patterns should emerge
    interaction2 = Interaction(
        interaction_id="test_2",
        session_id="test_session",
        user_query="Fix the authentication bug"
    )
    interaction2.actions_taken = completed.actions_taken  # Same actions
    interaction2.success = True
    
    patterns2 = await interaction_pattern_detector.detect_patterns([completed, interaction2])
    assert len(patterns2) > 0  # Now we should have patterns


@pytest.mark.asyncio
async def test_interaction_with_correction(interaction_logger, hybrid_store):
    """Test interaction that gets corrected by user."""
    # Start interaction
    interaction_id = interaction_logger.start_interaction("Update the configuration file")
    
    # AI misinterprets
    interaction_logger.log_interpretation(
        interpretation="User wants to update config.json",
        planned_actions=["edit_file"],
        confidence=0.7
    )
    
    # Take action
    action = interaction_logger.log_action("edit_file", target="config.json")
    interaction_logger.complete_action(action)
    
    # User corrects
    completed = interaction_logger.complete_interaction(
        feedback="No, I meant the .env configuration file",
        feedback_type=FeedbackType.CORRECTION,
        success=False
    )
    
    assert completed.was_corrected is True
    assert completed.success is False
    
    # Store and verify correction pattern can be found
    await hybrid_store.add_interaction(completed)
    corrections = await hybrid_store.find_correction_patterns()
    assert len(corrections) > 0


@pytest.mark.asyncio
async def test_pattern_learning_from_interactions(interaction_logger, hybrid_store, interaction_pattern_detector):
    """Test that patterns are learned from multiple similar interactions."""
    # Create 3 similar interactions
    for i in range(3):
        # Start
        interaction_logger.start_interaction(f"Fix bug in auth module #{i}")
        
        # Interpret
        interaction_logger.log_interpretation(
            "Fix authentication bug",
            ["search_files", "edit_file"],
            0.9
        )
        
        # Act
        action1 = interaction_logger.log_action("search_files", "auth")
        interaction_logger.complete_action(action1)
        
        action2 = interaction_logger.log_action("edit_file", "auth/token.py")
        interaction_logger.complete_action(action2)
        
        # Complete successfully
        completed = interaction_logger.complete_interaction(
            feedback="Fixed!",
            feedback_type=FeedbackType.APPROVAL,
            success=True
        )
        
        await hybrid_store.add_interaction(completed)
    
    # With random embeddings, we need to search for queries we know exist
    # Let's search for each of the queries we stored
    found_count = 0
    for i in range(3):
        similar = await hybrid_store.search_similar_queries(f"Fix bug in auth module #{i}", k=5)
        if len(similar) > 0:
            # Check the first result has the expected pattern
            for _, _, metadata in similar:
                if metadata.get("actions") == ["search_files", "edit_file"] and metadata.get("success") is True:
                    found_count += 1
                    break
    
    assert found_count >= 3, f"Expected to find 3 interactions with correct pattern, found {found_count}"


@pytest.mark.asyncio
async def test_query_suggestions(hybrid_store):
    """Test that similar queries provide action suggestions."""
    # Create an interaction
    interaction = Interaction(
        interaction_id="test_1",
        session_id="session_1",
        user_query="Add user registration feature",
        ai_interpretation="User wants to add registration functionality",
        planned_actions=["create_file", "edit_file", "add_tests"],
        success=True
    )
    
    # Add some actions
    interaction.actions_taken = [
        Action(action_type="create_file", target="auth/register.py", status=ActionStatus.COMPLETED),
        Action(action_type="edit_file", target="auth/routes.py", status=ActionStatus.COMPLETED),
        Action(action_type="add_tests", target="tests/test_register.py", status=ActionStatus.COMPLETED)
    ]
    
    interaction.user_feedback = "Perfect implementation!"
    interaction.feedback_type = FeedbackType.APPROVAL
    
    # Store it
    await hybrid_store.add_interaction(interaction)
    
    # Search for similar (with random embeddings, search for exact/similar text)
    similar = await hybrid_store.search_similar_queries("Add user registration feature", k=3)
    
    assert len(similar) > 0
    # The suggested actions should be available
    found = False
    for _, _, metadata in similar:
        if metadata.get("actions") == ["create_file", "edit_file", "add_tests"]:
            found = True
            break
    assert found, f"Expected actions not found in results: {similar}"