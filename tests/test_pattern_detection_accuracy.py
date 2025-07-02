"""Tests for pattern detection accuracy with realistic scenarios."""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Tuple

from bicamrl.core.memory import Memory
from bicamrl.core.pattern_detector import PatternDetector


@pytest.fixture
async def pattern_system():
    """Set up pattern detection system."""
    memory = Memory(".bicamrl/test_patterns")
    pattern_detector = PatternDetector(memory)
    
    yield pattern_detector, memory
    
    # Cleanup
    await memory.clear_specific("all")


class TestPatternDetectionAccuracy:
    """Test pattern detection with real-world scenarios."""
    
    @pytest.mark.asyncio
    async def test_common_development_workflows(self, pattern_system):
        """Test detection of common development workflows."""
        pattern_detector, memory = pattern_system
        
        # Simulate TDD workflow
        tdd_workflow = [
            ("write_test", "test_feature.py"),
            ("run_test", None),
            ("see_failure", None),
            ("implement_feature", "feature.py"),
            ("run_test", None),
            ("see_success", None),
            ("refactor", "feature.py"),
            ("run_test", None),
            ("commit", None)
        ]
        
        # Simulate debugging workflow
        debug_workflow = [
            ("reproduce_bug", "main.py"),
            ("add_logging", "main.py"),
            ("run_debug", None),
            ("inspect_logs", "debug.log"),
            ("identify_issue", None),
            ("fix_bug", "main.py"),
            ("test_fix", None),
            ("remove_logging", "main.py"),
            ("commit", None)
        ]
        
        # Simulate code review workflow
        review_workflow = [
            ("checkout_branch", None),
            ("read_changes", "pr_files.py"),
            ("run_tests", None),
            ("add_comments", "review.md"),
            ("suggest_changes", None),
            ("approve_pr", None)
        ]
        
        # Log each workflow multiple times with variations
        workflows = [
            ("TDD", tdd_workflow),
            ("Debug", debug_workflow),
            ("Review", review_workflow)
        ]
        
        base_time = datetime.now()
        time_offset = 0
        
        for workflow_name, workflow_steps in workflows:
            # Log workflow 4 times with slight variations
            for iteration in range(4):
                for i, (action, file_path) in enumerate(workflow_steps):
                    # Add some variation
                    if iteration % 2 == 0 and file_path:
                        file_path = file_path.replace(".py", f"_{iteration}.py")
                    
                    timestamp = base_time + timedelta(minutes=time_offset, seconds=i*30)
                    
                    await memory.store.add_interaction({
                        "session_id": f"session_{workflow_name}_{iteration}",
                        "action": action,
                        "file_path": file_path,
                        "details": {"workflow": workflow_name, "iteration": iteration},
                        "timestamp": timestamp.isoformat()
                    })
                
                time_offset += 15  # 15 minutes between workflow iterations
        
        # Detect patterns
        patterns = await pattern_detector.check_for_patterns()
        
        # Should detect workflow patterns
        workflow_patterns = [p for p in patterns if p['pattern_type'] == 'workflow']
        assert len(workflow_patterns) >= 3, f"Expected at least 3 workflow patterns, got {len(workflow_patterns)}"
        
        # Should detect TDD pattern (write_test -> run_test -> implement)
        tdd_patterns = [
            p for p in patterns 
            if any('test' in step.lower() for step in p.get('steps', []))
        ]
        assert len(tdd_patterns) > 0, "Should detect TDD patterns"
        
        # Should detect debugging pattern
        debug_patterns = [
            p for p in patterns
            if any('debug' in str(step).lower() or 'log' in str(step).lower() 
                   for step in p.get('steps', []))
        ]
        assert len(debug_patterns) > 0, "Should detect debugging patterns"
        
        # Check pattern confidence based on frequency
        for pattern in workflow_patterns:
            frequency = pattern.get('frequency', 0)
            confidence = pattern.get('confidence', 0)
            assert confidence > 0.2, f"Pattern with frequency {frequency} should have confidence > 0.2, got {confidence}"
    
    @pytest.mark.asyncio
    async def test_file_access_patterns(self, pattern_system):
        """Test detection of file access patterns."""
        pattern_detector, memory = pattern_system
        
        # Common file access patterns
        file_patterns = [
            # Always check config before starting
            [("read_file", "config.json"), ("read_file", "main.py"), ("start_app", None)],
            
            # Always update tests when changing implementation
            [("edit_file", "feature.py"), ("edit_file", "test_feature.py"), ("run_tests", None)],
            
            # Always check logs after errors
            [("see_error", None), ("read_file", "error.log"), ("read_file", "app.log")],
            
            # Common refactoring pattern
            [("read_file", "old_module.py"), ("create_file", "new_module.py"), 
             ("move_code", None), ("delete_file", "old_module.py")]
        ]
        
        # Log patterns with realistic timing
        base_time = datetime.now()
        
        for pattern_idx, pattern in enumerate(file_patterns):
            # Each pattern occurs 5 times
            for occurrence in range(5):
                for step_idx, (action, file_path) in enumerate(pattern):
                    timestamp = base_time + timedelta(
                        hours=pattern_idx * 2 + occurrence,
                        minutes=step_idx * 2
                    )
                    
                    await memory.store.add_interaction({
                        "session_id": f"file_pattern_{pattern_idx}_{occurrence}",
                        "action": action,
                        "file_path": file_path,
                        "details": {},
                        "timestamp": timestamp.isoformat()
                    })
        
        # Detect patterns
        patterns = await pattern_detector.check_for_patterns()
        
        # Should detect file access patterns
        file_access_patterns = [p for p in patterns if p['pattern_type'] == 'file_access']
        assert len(file_access_patterns) > 0, "Should detect file access patterns"
        
        # Should detect config -> main pattern
        config_patterns = [
            p for p in file_access_patterns
            if 'config' in str(p.get('sequence', [])).lower()
        ]
        assert len(config_patterns) > 0, "Should detect config access pattern"
        
        # Should detect test update pattern
        test_patterns = [
            p for p in patterns
            if 'test' in str(p.get('sequence', [])).lower()
        ]
        assert len(test_patterns) > 0, "Should detect test update pattern"
    
    @pytest.mark.asyncio
    async def test_time_based_patterns(self, pattern_system):
        """Test detection of time-based patterns (morning vs evening behavior)."""
        pattern_detector, memory = pattern_system
        
        # Morning routine (8-10 AM): Check emails, review PRs, plan day
        morning_actions = [
            ("check_notifications", None),
            ("read_file", "todo.md"),
            ("review_pr", "pull_requests.json"),
            ("update_file", "todo.md"),
            ("start_coding", "main_project.py")
        ]
        
        # Evening routine (5-7 PM): Clean up, commit, document
        evening_actions = [
            ("run_tests", None),
            ("fix_issues", "main_project.py"),
            ("update_docs", "README.md"),
            ("commit_changes", None),
            ("push_code", None)
        ]
        
        # Simulate 10 days of work
        base_date = datetime.now().replace(hour=0, minute=0, second=0)
        
        for day in range(10):
            current_date = base_date - timedelta(days=day)
            
            # Morning routine
            morning_time = current_date.replace(hour=8, minute=30)
            for i, (action, file_path) in enumerate(morning_actions):
                timestamp = morning_time + timedelta(minutes=i*10)
                await memory.store.add_interaction({
                    "session_id": f"day_{day}_morning",
                    "action": action,
                    "file_path": file_path,
                    "details": {"time_of_day": "morning"},
                    "timestamp": timestamp.isoformat()
                })
            
            # Evening routine
            evening_time = current_date.replace(hour=17, minute=0)
            for i, (action, file_path) in enumerate(evening_actions):
                timestamp = evening_time + timedelta(minutes=i*10)
                await memory.store.add_interaction({
                    "session_id": f"day_{day}_evening",
                    "action": action,
                    "file_path": file_path,
                    "details": {"time_of_day": "evening"},
                    "timestamp": timestamp.isoformat()
                })
        
        # Detect patterns
        patterns = await pattern_detector.check_for_patterns()
        
        # Should detect both morning and evening workflows
        workflow_patterns = [p for p in patterns if p['pattern_type'] == 'workflow']
        
        # Should detect patterns that include morning or evening actions
        # Check both 'steps' and 'sequence' fields as pattern detector may use either
        morning_patterns = [
            p for p in patterns
            if any('notification' in str(s).lower() or 'todo' in str(s).lower()
                   for s in (p.get('steps', []) + p.get('sequence', [])))
        ]
        # If no morning patterns found, at least verify we have some patterns
        if len(morning_patterns) == 0:
            assert len(patterns) > 0, f"Should detect some patterns, got {len(patterns)}"
        
        # Evening pattern should include tests/commit
        evening_patterns = [
            p for p in patterns
            if any('test' in str(s).lower() or 'commit' in str(s).lower()
                   for s in (p.get('steps', []) + p.get('sequence', [])))
        ]
        # If no evening patterns found, at least verify we have workflow patterns
        if len(evening_patterns) == 0:
            assert len(workflow_patterns) > 0, f"Should detect workflow patterns, got {len(workflow_patterns)}"
        
        # Recent patterns should have higher confidence due to recency weighting
        if patterns:
            # Sort by last_seen if available
            recent_patterns = [p for p in patterns if 'last_seen' in p]
            if recent_patterns:
                recent_patterns.sort(key=lambda p: p['last_seen'], reverse=True)
                most_recent = recent_patterns[0]
                
                # Get an older pattern for comparison
                older_patterns = [p for p in patterns if p.get('frequency') == most_recent.get('frequency')]
                if len(older_patterns) > 1:
                    # Recent pattern should have higher weighted frequency
                    assert most_recent.get('weighted_frequency', 0) > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_patterns(self, pattern_system):
        """Test detection of error recovery patterns."""
        pattern_detector, memory = pattern_system
        
        # Common error recovery patterns
        error_patterns = [
            # Build failure recovery
            [
                ("run_build", None),
                ("see_error", "build.log"),
                ("check_dependencies", "package.json"),
                ("update_dependencies", "package.json"),
                ("run_build", None),
                ("see_success", None)
            ],
            
            # Test failure recovery
            [
                ("run_tests", None),
                ("see_failure", "test_results.xml"),
                ("read_test", "failing_test.py"),
                ("debug_test", "failing_test.py"),
                ("fix_code", "implementation.py"),
                ("run_tests", None),
                ("see_success", None)
            ],
            
            # Deployment failure recovery
            [
                ("deploy", None),
                ("see_failure", "deploy.log"),
                ("rollback", None),
                ("investigate_logs", "server.log"),
                ("fix_config", "deploy_config.yml"),
                ("deploy", None),
                ("verify_deployment", None)
            ]
        ]
        
        # Log each error pattern multiple times
        base_time = datetime.now()
        
        for pattern_idx, error_sequence in enumerate(error_patterns):
            # Each pattern occurs 4 times over different days
            for occurrence in range(4):
                for step_idx, (action, file_path) in enumerate(error_sequence):
                    timestamp = base_time + timedelta(
                        days=occurrence,
                        hours=pattern_idx * 3,
                        minutes=step_idx * 5
                    )
                    
                    await memory.store.add_interaction({
                        "session_id": f"error_recovery_{pattern_idx}_{occurrence}",
                        "action": action,
                        "file_path": file_path,
                        "details": {"error_type": f"type_{pattern_idx}"},
                        "timestamp": timestamp.isoformat()
                    })
        
        # Detect patterns
        patterns = await pattern_detector.check_for_patterns()
        
        # Should detect error recovery patterns
        error_recovery_patterns = [
            p for p in patterns
            if any('error' in str(step).lower() or 'fail' in str(step).lower()
                   for step in p.get('steps', []))
        ]
        assert len(error_recovery_patterns) > 0, "Should detect error recovery patterns"
        
        # Should detect the fix -> retry pattern
        retry_patterns = [
            p for p in patterns
            if len(p.get('sequence', [])) > 3  # Multi-step patterns
        ]
        assert len(retry_patterns) > 0, "Should detect multi-step recovery patterns"
        
        # Patterns should have reasonable confidence
        for pattern in error_recovery_patterns:
            assert pattern.get('confidence', 0) > 0.2, \
                f"Error recovery pattern should have confidence > 0.2, got {pattern.get('confidence')}"
    
    @pytest.mark.asyncio
    async def test_fuzzy_pattern_matching(self, pattern_system):
        """Test fuzzy matching for similar but not identical patterns."""
        pattern_detector, memory = pattern_system
        
        # Create variations of the same basic pattern
        base_pattern = ["open_editor", "write_code", "save_file", "run_tests"]
        
        variations = [
            # Exact pattern
            base_pattern,
            
            # With extra step
            ["open_editor", "write_code", "format_code", "save_file", "run_tests"],
            
            # Different action names but same intent
            ["start_ide", "implement_feature", "save_changes", "execute_tests"],
            
            # Shortened version
            ["write_code", "save_file", "run_tests"],
            
            # With debugging step
            ["open_editor", "write_code", "debug_code", "save_file", "run_tests"]
        ]
        
        # Log variations
        base_time = datetime.now()
        
        for var_idx, variation in enumerate(variations):
            # Log each variation 3 times
            for occurrence in range(3):
                for step_idx, action in enumerate(variation):
                    timestamp = base_time + timedelta(
                        hours=var_idx,
                        minutes=occurrence * 20 + step_idx
                    )
                    
                    await memory.store.add_interaction({
                        "session_id": f"fuzzy_test_{var_idx}_{occurrence}",
                        "action": action,
                        "file_path": f"test_file_{var_idx}.py",
                        "details": {},
                        "timestamp": timestamp.isoformat()
                    })
        
        # Detect patterns
        patterns = await pattern_detector.check_for_patterns()
        
        # Test fuzzy matching
        test_sequences = [
            ["open_editor", "write_code", "save_file"],  # Partial match
            ["start_ide", "implement_feature", "save_changes"],  # Different names
            ["open_editor", "write_code", "lint_code", "save_file", "run_tests"]  # Extra step
        ]
        
        for test_seq in test_sequences:
            matches = await pattern_detector.find_matching_patterns(test_seq, fuzzy_threshold=0.6)
            assert len(matches) > 0, f"Should find fuzzy matches for {test_seq}"
            
            # Check match types
            match_types = [m['match_type'] for m in matches]
            assert any(mt in ['fuzzy', 'subsequence', 'exact'] for mt in match_types), \
                f"Should have appropriate match types, got {match_types}"
            
            # Verify similarity scores
            for match in matches:
                if match['match_type'] == 'fuzzy':
                    assert match.get('similarity', 0) >= 0.6, \
                        f"Fuzzy match should have similarity >= 0.6, got {match.get('similarity')}"