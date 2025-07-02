"""Test semantic extraction functionality."""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from bicamrl.core.memory import Memory
from bicamrl.core.memory_consolidator import MemoryConsolidator
from bicamrl.sleep.llm_providers import MockLLMProvider


class SemanticMockLLMProvider(MockLLMProvider):
    """Mock LLM provider that returns semantic analysis."""
    
    async def complete(self, prompt: str, max_tokens: int = 100) -> str:
        """Return semantic analysis based on prompt content."""
        if "work session" in prompt.lower():
            return "Summary: Developer fixed authentication bugs and added input validation. Primary goal was improving security. Detected TDD workflow pattern. Next step: Add integration tests."
        
        elif "episode" in prompt.lower():
            return "Narrative: A focused debugging session where the developer systematically tracked down a race condition in the payment processing module. Key learning: Always use mutexes for payment operations. Success: Bug fixed and tests added."
        
        elif "recurring pattern" in prompt.lower():
            return "Principle: This test-first pattern ensures code correctness before implementation. Apply when adding new features or fixing bugs. Effective because it catches issues early. Improvement: Add property-based tests for edge cases."
        
        elif "preferences" in prompt.lower():
            return "Core principles: Clean, testable code with explicit error handling. Philosophy: Fail fast and provide clear error messages. These preferences create maintainable, debuggable code. Recommendation: Apply these patterns to all new modules."
        
        elif "holistically" in prompt.lower():
            return "Workflow style: Methodical, test-driven developer who values code quality. Common themes: Testing, validation, error handling. Strengths: Thorough testing, good documentation. Improvement area: Could benefit from more automated tooling. Best practice: Continue TDD approach but add CI/CD automation."
        
        return "Generic semantic analysis response"


@pytest.fixture
async def semantic_memory():
    """Create memory with semantic extraction capability."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(tmpdir)
        llm_provider = SemanticMockLLMProvider()
        memory.set_llm_provider(llm_provider)
        yield memory


@pytest.mark.asyncio
async def test_work_session_semantic_extraction(semantic_memory):
    """Test semantic extraction for work sessions."""
    # Create some test interactions
    base_time = datetime.now() - timedelta(hours=2)
    
    interactions = []
    for i in range(5):
        interaction = {
            'timestamp': (base_time + timedelta(minutes=i*10)).isoformat(),
            'session_id': 'test_session',
            'action': ['read_file', 'edit_file', 'run_tests', 'commit', 'push'][i],
            'file_path': 'auth/login.py',
            'details': {'line': i*10, 'change': f'fix_{i}'}
        }
        await semantic_memory.store.add_interaction(interaction)
        interactions.append(interaction)
    
    # Create work summary
    consolidator = semantic_memory.consolidator
    summary = await consolidator._create_work_summary(interactions)
    
    assert summary is not None
    assert summary['has_semantic_analysis'] is True
    assert 'semantic_summary' in summary
    assert 'authentication' in summary['semantic_summary'].lower()
    assert 'security' in summary['semantic_summary'].lower()


@pytest.mark.asyncio
async def test_episode_semantic_extraction(semantic_memory):
    """Test semantic extraction for episodes."""
    # Create test interactions for an episode
    base_time = datetime.now() - timedelta(days=2)
    
    interactions = []
    for i in range(10):
        interaction = {
            'timestamp': (base_time + timedelta(minutes=i*5)).isoformat(),
            'session_id': 'episode_session',
            'action': 'debug' if i < 5 else 'fix',
            'file_path': f'payment/module_{i%3}.py',
            'details': {'issue': 'race_condition', 'step': i}
        }
        await semantic_memory.store.add_interaction(interaction)
        interactions.append(interaction)
    
    # Create episode summary
    consolidator = semantic_memory.consolidator
    summary = await consolidator._create_episode_summary(interactions)
    
    assert summary is not None
    assert summary['type'] == 'episode'
    assert summary.get('has_episode_analysis') is True
    assert 'episode_narrative' in summary
    assert 'race condition' in summary['episode_narrative'].lower()
    assert 'payment' in summary['episode_narrative'].lower()


@pytest.mark.asyncio
async def test_semantic_knowledge_extraction(semantic_memory):
    """Test extraction of semantic knowledge from patterns."""
    # Add some patterns
    patterns = [
        {
            'name': 'TDD Workflow',
            'description': 'Test-driven development pattern',
            'pattern_type': 'workflow',
            'frequency': 10,
            'confidence': 0.9,
            'sequence': ['write_test', 'run_test', 'implement', 'run_test']
        },
        {
            'name': 'Error Handling Pattern',
            'description': 'Consistent error handling',
            'pattern_type': 'code_style',
            'frequency': 8,
            'confidence': 0.85,
            'sequence': ['validate', 'try', 'catch', 'log']
        }
    ]
    
    for pattern in patterns:
        await semantic_memory.store.add_pattern(pattern)
    
    # Add preferences - need more than 2 per category for synthesis
    preferences = [
        ('error_format', 'structured_json', 'error_handling'),
        ('error_logging', 'with_context', 'error_handling'),
        ('error_recovery', 'graceful_degradation', 'error_handling'),
        ('log_level', 'debug_in_dev', 'logging'),
        ('log_format', 'json_structured', 'logging'),
        ('log_destination', 'centralized', 'logging'),
        ('test_framework', 'pytest', 'testing'),
        ('assertion_style', 'explicit_messages', 'testing'),
        ('test_isolation', 'complete', 'testing'),
        ('test_naming', 'descriptive_behavior', 'testing')
    ]
    
    for key, value, category in preferences:
        await semantic_memory.store.add_preference({
            'key': key,
            'value': value,
            'category': category
        })
    
    # Run semantic extraction
    consolidator = semantic_memory.consolidator
    count = await consolidator._extract_semantic_knowledge([])
    
    assert count > 0
    
    # Check that semantic knowledge was stored
    all_patterns = await semantic_memory.get_all_patterns()
    semantic_patterns = [p for p in all_patterns if 'consolidated_semantic' in p.get('pattern_type', '')]
    
    assert len(semantic_patterns) > 0
    
    # Check for different types of semantic knowledge
    has_pattern_principle = False
    has_preference_synthesis = False
    has_meta_analysis = False
    
    for sp in semantic_patterns:
        metadata = sp.get('metadata', {})
        if metadata.get('type') == 'semantic_pattern' and metadata.get('has_semantic_extraction'):
            has_pattern_principle = True
            assert 'semantic_principle' in metadata
            
        elif metadata.get('type') == 'semantic_preference':
            if metadata.get('has_synthesis'):
                has_preference_synthesis = True
                assert 'synthesized_principles' in metadata
            
        elif metadata.get('type') == 'semantic_meta':
            has_meta_analysis = True
            assert 'key_attributes' in metadata
            assert 'analysis' in metadata['key_attributes']
    
    assert has_pattern_principle, "Should have extracted pattern principles"
    assert has_preference_synthesis, "Should have synthesized preferences"


@pytest.mark.asyncio
async def test_memory_consolidation_with_semantics(semantic_memory):
    """Test full memory consolidation with semantic extraction."""
    # Create interactions at different time periods
    now = datetime.now()
    
    # Old interactions for semantic extraction
    for i in range(20):
        old_time = now - timedelta(days=10 + i)
        await semantic_memory.store.add_interaction({
            'timestamp': old_time.isoformat(),
            'session_id': f'old_session_{i//5}',
            'action': ['read', 'edit', 'test', 'commit'][i % 4],
            'file_path': f'module_{i % 3}.py',
            'details': {'work': f'task_{i}'}
        })
    
    # Working memory interactions
    for i in range(10):
        work_time = now - timedelta(hours=12 + i)
        await semantic_memory.store.add_interaction({
            'timestamp': work_time.isoformat(),
            'session_id': 'work_session',
            'action': 'implement',
            'file_path': 'feature.py',
            'details': {'feature': f'step_{i}'}
        })
    
    # Run consolidation
    stats = await semantic_memory.consolidate_memories()
    
    assert stats['active_to_working'] > 0
    assert stats['episodic_to_semantic'] >= 0
    
    # Check consolidated memories
    consolidated = await semantic_memory.get_consolidated_memories()
    assert len(consolidated) > 0
    
    # Verify semantic content in consolidated memories
    has_semantic_content = False
    for memory in consolidated:
        if memory['data'].get('has_semantic_analysis') or memory['data'].get('has_semantic_extraction'):
            has_semantic_content = True
            break
    
    assert has_semantic_content, "Consolidated memories should have semantic content"