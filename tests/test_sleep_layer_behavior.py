"""Tests for Sleep Layer behavior."""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from bicamrl.sleep.sleep import (
    Sleep,
    Observation,
    Insight,
    AnalysisType
)
from bicamrl.sleep.llm_providers import MockLLMProvider
from bicamrl.core.memory import Memory


@pytest.fixture
async def sleep_system():
    """Create Sleep Layer with mock providers."""
    memory = Memory(".bicamrl/test_sleep")

    # Create mock LLM providers
    providers = {
        'analyzer': MockLLMProvider(),
        'generator': MockLLMProvider(),
        'enhancer': MockLLMProvider(),
        'optimizer': MockLLMProvider()
    }

    config = {
        'batch_size': 3,
        'analysis_interval': 1,  # 1 second for testing
        'min_confidence': 0.6
    }

    sleep_layer = Sleep(memory, providers, config)

    yield sleep_layer, memory

    # Cleanup
    if sleep_layer.is_running:
        await sleep_layer.stop()
    await memory.clear_specific("all")


class TestSleepBehavior:
    """Test Sleep Layer meta-cognitive behaviors."""

    @pytest.mark.asyncio
    async def test_observation_processing(self, sleep_layer_system):
        """Test that Sleep Layer processes observations correctly."""
        sleep_layer, memory = sleep_layer_system

        await sleep_layer.start()

        # Create test observations
        observations = [
            Observation(
                timestamp=datetime.now(),
                interaction_type="code_edit",
                query="Fix the bug in authentication",
                context_used={"file": "auth.py"},
                response="Fixed null check",
                tokens_used=150,
                latency=0.5,
                success=True
            ),
            Observation(
                timestamp=datetime.now(),
                interaction_type="test_run",
                query="Run auth tests",
                context_used={"file": "test_auth.py"},
                response="All tests passed",
                tokens_used=100,
                latency=0.3,
                success=True
            ),
            Observation(
                timestamp=datetime.now(),
                interaction_type="code_review",
                query="Review auth changes",
                context_used={"files": ["auth.py", "test_auth.py"]},
                response="Approved",
                tokens_used=200,
                latency=0.8,
                success=True
            )
        ]

        # Add observations
        for obs in observations:
            await sleep_layer.observe(obs)

        # Wait for batch processing
        await asyncio.sleep(0.2)

        # Check that observations were logged
        interactions = await memory.store.get_recent_interactions(10)
        assert len(interactions) >= 3

        # Verify interaction details
        assert any(i['action'] == 'code_edit' for i in interactions)
        assert any(i['action'] == 'test_run' for i in interactions)

    @pytest.mark.asyncio
    async def test_critical_observation_handling(self, sleep_layer_system):
        """Test immediate analysis of critical observations."""
        sleep_layer, _ = sleep_layer_system

        await sleep_layer.start()

        # Create critical observations
        high_latency_obs = Observation(
            timestamp=datetime.now(),
            interaction_type="query",
            query="Complex analysis",
            context_used={},
            response="Result",
            tokens_used=500,
            latency=15.0,  # High latency
            success=True
        )

        failure_obs = Observation(
            timestamp=datetime.now(),
            interaction_type="code_gen",
            query="Generate function",
            context_used={},
            response="Error: syntax error",
            tokens_used=100,
            latency=1.0,
            success=False  # Failure
        )

        high_token_obs = Observation(
            timestamp=datetime.now(),
            interaction_type="analysis",
            query="Analyze codebase",
            context_used={},
            response="Analysis complete",
            tokens_used=15000,  # High token usage
            latency=5.0,
            success=True
        )

        # Track insights before
        insights_before = len(sleep_layer.insights_cache)

        # Add critical observations
        await sleep_layer.observe(high_latency_obs)
        await sleep_layer.observe(failure_obs)
        await sleep_layer.observe(high_token_obs)

        # Give time for immediate analysis
        await asyncio.sleep(0.5)

        # Should have generated insights
        insights_after = len(sleep_layer.insights_cache)
        assert insights_after > insights_before

        # Check insight types
        recent_insights = sleep_layer.insights_cache[insights_before:]
        assert any(i.type == AnalysisType.CONTEXT_OPTIMIZATION for i in recent_insights)

    @pytest.mark.asyncio
    async def test_pattern_mining_insights(self, sleep_layer_system):
        """Test pattern mining generates useful insights."""
        sleep_layer, memory = sleep_layer_system

        # Pre-populate some patterns
        test_patterns = [
            {
                'name': 'Edit-Test-Commit',
                'description': 'Common development workflow',
                'pattern_type': 'workflow',
                'sequence': ['edit', 'test', 'commit'],
                'frequency': 10,
                'confidence': 0.9
            },
            {
                'name': 'Debug-Fix-Test',
                'description': 'Bug fixing workflow',
                'pattern_type': 'workflow',
                'sequence': ['debug', 'fix', 'test'],
                'frequency': 5,
                'confidence': 0.8
            }
        ]

        for pattern in test_patterns:
            await memory.store.add_pattern(pattern)

        # Mock analyzer to return specific insights
        mock_response = json.dumps({
            "valuable_patterns": [
                {
                    "name": "Edit-Test-Commit",
                    "recommendation": "Consider creating a git hook for this workflow"
                }
            ],
            "automation_opportunities": [
                {
                    "description": "Auto-run tests before commit",
                    "steps": ["Install pre-commit", "Configure test runner"]
                }
            ]
        })

        sleep_layer.llms['analyzer'].analyze = AsyncMock(return_value=mock_response)

        await sleep_layer.start()

        # Trigger periodic analysis
        await sleep_layer._periodic_analyzer()

        # Check insights were generated
        assert len(sleep_layer.insights_cache) > 0

        # Verify insight content
        pattern_insights = [
            i for i in sleep_layer.insights_cache
            if i.type == AnalysisType.PATTERN_MINING
        ]
        assert len(pattern_insights) > 0

        # Check for automation recommendations
        automation_insights = [
            i for i in pattern_insights
            if 'automation' in i.description.lower()
        ]
        assert len(automation_insights) > 0

    @pytest.mark.asyncio
    async def test_insight_application(self, sleep_layer_system):
        """Test that insights are applied to improve the system."""
        sleep_layer, memory = sleep_layer_system

        await sleep_layer.start()

        # Create test insights
        pattern_insight = Insight(
            type=AnalysisType.PATTERN_MINING,
            confidence=0.9,
            description="Valuable pattern discovered",
            recommendations=["Use this pattern more"],
            data={
                'type': 'workflow',
                'description': 'Efficient testing workflow'
            }
        )

        context_insight = Insight(
            type=AnalysisType.CONTEXT_OPTIMIZATION,
            confidence=0.85,
            description="Context optimization",
            recommendations=["Always include: config.py, utils.py"],
            data={}
        )

        error_insight = Insight(
            type=AnalysisType.ERROR_ANALYSIS,
            confidence=0.8,
            description="Common error pattern",
            recommendations=["Check null values"],
            data={'error_type': 'NullPointerException'}
        )

        # Apply insights
        await sleep_layer._apply_insight(pattern_insight)
        await sleep_layer._apply_insight(context_insight)
        await sleep_layer._apply_insight(error_insight)

        # Verify insights were applied

        # Pattern should be stored
        patterns = await memory.get_all_patterns()
        sleep_layer_patterns = [p for p in patterns if p.get('source') == 'sleep_layer_analysis']
        assert len(sleep_layer_patterns) > 0

        # Context preference should be stored
        preferences = await memory.get_preferences()
        context_prefs = preferences.get('context', {})
        assert 'always_include_files' in context_prefs

        # Error pattern should be stored
        error_patterns = [p for p in patterns if p.get('pattern_type') == 'error']
        assert len(error_patterns) > 0

    @pytest.mark.asyncio
    async def test_prompt_enhancement(self, sleep_layer_system):
        """Test Sleep Layer enhances prompts based on learning."""
        sleep_layer, memory = sleep_layer_system

        # Add some preferences and patterns
        await memory.store.add_preference({
            'key': 'code_style',
            'value': 'Use type hints',
            'category': 'style',
            'confidence': 0.9
        })

        await memory.store.add_pattern({
            'name': 'Test-first development',
            'description': 'Write tests before implementation',
            'pattern_type': 'workflow',
            'sequence': ['write_test', 'implement', 'refactor'],
            'confidence': 0.85
        })

        # Mock enhancer response
        mock_enhancement = json.dumps({
            "enhanced_query": "Implement user authentication with type hints, following test-first development",
            "relevant_context": ["auth.py", "test_auth.py"],
            "examples": ["def authenticate(username: str, password: str) -> bool:"],
            "output_format": "Include docstrings and type annotations"
        })

        sleep_layer.llms['enhancer'].generate = AsyncMock(return_value=mock_enhancement)

        await sleep_layer.start()

        # Test prompt enhancement
        original_query = "Implement user authentication"
        recommendation = await sleep_layer.get_prompt_recommendation(
            query=original_query,
            current_context={'files': ['main.py']}
        )

        assert 'enhanced_query' in recommendation
        assert 'type hints' in recommendation['enhanced_query']
        assert 'relevant_context' in recommendation

    @pytest.mark.asyncio
    async def test_multi_llm_coordination(self, sleep_layer_system):
        """Test coordination between multiple LLM providers."""
        sleep_layer, _ = sleep_layer_system

        # Create different mock responses for different providers
        analyzer_response = json.dumps({
            "patterns": [{"type": "performance", "description": "Slow query"}]
        })

        optimizer_response = "Optimize by adding database index"

        sleep_layer.llms['analyzer'].analyze = AsyncMock(return_value=analyzer_response)
        sleep_layer.llms['optimizer'].analyze = AsyncMock(return_value=optimizer_response)

        await sleep_layer.start()

        # Create performance issue observation
        slow_query_obs = Observation(
            timestamp=datetime.now(),
            interaction_type="database_query",
            query="SELECT * FROM users WHERE email = ?",
            context_used={"table": "users"},
            response="Query took 5 seconds",
            tokens_used=50,
            latency=5.0,
            success=True
        )

        await sleep_layer.observe(slow_query_obs)

        # Wait for analysis
        await asyncio.sleep(0.5)

        # Both providers should have been called
        sleep_layer.llms['analyzer'].analyze.assert_called()

        # Context optimization should use optimizer
        await sleep_layer._optimize_context({'top_files': [{'file': 'db.py', 'count': 10}]})
        sleep_layer.llms['optimizer'].analyze.assert_called()

    @pytest.mark.asyncio
    async def test_learning_from_failures(self, sleep_layer_system):
        """Test Sleep Layer learns from failures to prevent repetition."""
        sleep_layer, memory = sleep_layer_system

        # Mock analyzer for failure analysis
        failure_analysis = """
        Root cause: Missing null check before accessing user.email
        Prevention: Always validate user object exists before accessing properties
        """

        sleep_layer.llms['analyzer'].analyze = AsyncMock(return_value=failure_analysis)

        await sleep_layer.start()

        # Report a failure
        failure_obs = Observation(
            timestamp=datetime.now(),
            interaction_type="code_execution",
            query="Send email to user",
            context_used={"function": "send_email"},
            response="Error: Cannot read property 'email' of null",
            tokens_used=100,
            latency=0.5,
            success=False,
            metadata={"error_type": "NullError"}
        )

        await sleep_layer.observe(failure_obs)

        # Wait for analysis
        await asyncio.sleep(0.5)

        # Should have created error analysis insight
        error_insights = [
            i for i in sleep_layer.insights_cache
            if i.type == AnalysisType.ERROR_ANALYSIS
        ]
        assert len(error_insights) > 0

        # Insight should have been applied
        patterns = await memory.get_all_patterns()
        error_patterns = [p for p in patterns if p.get('pattern_type') == 'error']
        assert len(error_patterns) > 0

        # Future similar queries should benefit from this learning
        context = await memory.get_relevant_context(
            task_description="send notification to user",
            file_context=["notifications.py"]
        )

        # Should include error patterns in context
        assert any('null' in str(p).lower() for p in context.get('relevant_patterns', []))

    @pytest.mark.asyncio
    async def test_adaptive_batch_processing(self, sleep_layer_system):
        """Test Sleep Layer adapts batch processing based on load."""
        sleep_layer, _ = sleep_layer_system

        sleep_layer.batch_size = 3  # Small batch for testing
        await sleep_layer.start()

        # Send observations in bursts
        burst_size = 10
        for i in range(burst_size):
            obs = Observation(
                timestamp=datetime.now(),
                interaction_type="action",
                query=f"Query {i}",
                context_used={},
                response=f"Response {i}",
                tokens_used=50,
                latency=0.1,
                success=True
            )
            await sleep_layer.observe(obs)

        # Wait for processing
        await asyncio.sleep(1)

        # All observations should be processed
        assert sleep_layer.observation_queue.qsize() == 0

        # Send critical observation during normal processing
        critical_obs = Observation(
            timestamp=datetime.now(),
            interaction_type="critical",
            query="Critical operation",
            context_used={},
            response="Failed",
            tokens_used=100,
            latency=0.5,
            success=False
        )

        await sleep_layer.observe(critical_obs)

        # Critical observation should trigger immediate processing
        await asyncio.sleep(0.2)

        # Should have insights from critical observation
        assert any(
            i.data.get('observation', {}).get('interaction_type') == 'critical'
            for i in sleep_layer.insights_cache
            if hasattr(i, 'data') and i.data
        )
