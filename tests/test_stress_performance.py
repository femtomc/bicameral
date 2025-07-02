"""Stress tests and performance benchmarks for Bicamrl system."""

import asyncio
import random
import statistics
import string
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

from bicamrl.core.feedback_processor import FeedbackProcessor
from bicamrl.core.interaction_logger import InteractionLogger
from bicamrl.core.interaction_pattern_detector import InteractionPatternDetector
from bicamrl.core.memory import Memory
from bicamrl.core.pattern_detector import PatternDetector
from bicamrl.storage.hybrid_store import HybridStore


@pytest.fixture
async def stress_test_system():
    """Set up system for stress testing."""
    memory = Memory(".bicamrl/stress_test")
    pattern_detector = PatternDetector(memory)
    feedback_processor = FeedbackProcessor(memory)
    interaction_logger = InteractionLogger(memory)
    interaction_pattern_detector = InteractionPatternDetector(memory)
    hybrid_store = HybridStore(Path(".bicamrl/stress_test").parent)

    yield (
        memory,
        pattern_detector,
        feedback_processor,
        interaction_logger,
        interaction_pattern_detector,
        hybrid_store,
    )

    # Cleanup
    await memory.clear_specific("all")


class TestStressPerformance:
    """Stress tests and performance benchmarks."""

    @pytest.mark.asyncio
    async def test_high_volume_interactions(self, stress_test_system):
        """Test system with high volume of interactions."""
        (
            memory,
            pattern_detector,
            _,
            interaction_logger,
            interaction_pattern_detector,
            hybrid_store,
        ) = stress_test_system

        # Performance targets
        target_interactions = 10000
        target_write_time = 10.0  # seconds for all writes
        target_read_time = 1.0  # seconds for retrieval

        # Generate test data
        actions = ["edit", "save", "test", "commit", "review", "debug", "refactor"]
        files = [f"file_{i}.py" for i in range(100)]

        start_time = time.time()

        # Write interactions using new API
        interactions_created = 0
        for batch in range(target_interactions // 100):
            # Create interactions in batches
            batch_tasks = []
            for i in range(100):
                idx = batch * 100 + i
                if idx >= target_interactions:
                    break

                action = random.choice(actions)
                file_path = random.choice(files) if random.random() > 0.2 else None

                # Use new interaction API
                interaction_id = interaction_logger.start_interaction(
                    user_query=f"Perform {action} on files"
                )
                interaction_logger.log_action(
                    action_type=action,
                    target=file_path,
                    details={"index": idx, "batch": idx // 1000},
                )
                completed = interaction_logger.complete_interaction(success=True)

                if completed:
                    batch_tasks.append(hybrid_store.add_interaction(completed))
                    interactions_created += 1

            # Process batch
            if batch_tasks:
                await asyncio.gather(*batch_tasks)

        # Verify interactions were created
        assert interactions_created > 0, "No interactions were created"

        write_time = time.time() - start_time

        # Test write performance
        assert write_time < target_write_time, (
            f"Writing {target_interactions} interactions took {write_time:.2f}s, target was {target_write_time}s"
        )

        # Test read performance
        start_time = time.time()
        recent_interactions = await memory.store.get_recent_interactions(1000)
        read_time = time.time() - start_time

        assert read_time < target_read_time, (
            f"Reading 1000 interactions took {read_time:.2f}s, target was {target_read_time}s"
        )

        assert len(recent_interactions) == 1000

        # Test pattern detection performance with new detector
        start_time = time.time()
        # Get recent interactions for pattern detection
        recent_interactions_data = await memory.store.get_recent_interactions(1000)
        # Convert to Interaction objects (simplified for test)
        from bicamrl.core.interaction_model import Action, Interaction

        interactions = []
        for data in recent_interactions_data[:100]:  # Use subset for performance
            interaction = Interaction(user_query="test")
            if "action" in data:
                action = Action(action_type=data["action"], target=data.get("file_path"))
                interaction.actions_taken.append(action)
            interactions.append(interaction)

        patterns = await interaction_pattern_detector.detect_patterns(interactions)
        pattern_time = time.time() - start_time

        assert pattern_time < 5.0, f"Pattern detection took {pattern_time:.2f}s, should be < 5s"

        # Calculate throughput
        throughput = interactions_created / write_time
        print(f"\nPerformance metrics:")
        print(f"  Write throughput: {throughput:.0f} interactions/second")
        print(f"  Read latency: {read_time * 1000:.1f}ms for 1000 records")
        print(f"  Pattern detection: {pattern_time:.2f}s for {len(patterns)} patterns")

    @pytest.mark.asyncio
    async def test_concurrent_access_stress(self, stress_test_system):
        """Test system under heavy concurrent access."""
        (
            memory,
            pattern_detector,
            _,
            interaction_logger,
            interaction_pattern_detector,
            hybrid_store,
        ) = stress_test_system

        num_concurrent_users = 50
        interactions_per_user = 100

        async def simulate_user(user_id: int) -> Dict[str, float]:
            """Simulate a user with mixed operations."""
            timings = []

            for i in range(interactions_per_user):
                operation_start = time.time()

                # Mix of operations
                operation = random.choice(["write", "read", "search", "pattern"])

                if operation == "write":
                    # Use new interaction API
                    interaction_id = interaction_logger.start_interaction(
                        user_query=f"User {user_id} action {i}"
                    )
                    interaction_logger.log_action(
                        action_type=f"action_{i}", target=f"user_{user_id}_file_{i}.py"
                    )
                    completed = interaction_logger.complete_interaction(success=True)
                    if completed:
                        await hybrid_store.add_interaction(completed)
                elif operation == "read":
                    await memory.get_recent_context(limit=20)
                elif operation == "search":
                    await memory.search(f"user_{user_id}")
                elif operation == "pattern":
                    await pattern_detector.find_matching_patterns(["action_1", "action_2"])

                operation_time = time.time() - operation_start
                timings.append(operation_time)

                # Small delay to simulate real usage
                await asyncio.sleep(random.uniform(0.01, 0.05))

            return {
                "user_id": user_id,
                "avg_latency": statistics.mean(timings),
                "p95_latency": statistics.quantiles(timings, n=20)[18],  # 95th percentile
                "max_latency": max(timings),
            }

        # Run concurrent users
        start_time = time.time()
        user_tasks = [simulate_user(i) for i in range(num_concurrent_users)]
        results = await asyncio.gather(*user_tasks)
        total_time = time.time() - start_time

        # Analyze results
        avg_latencies = [r["avg_latency"] for r in results]
        p95_latencies = [r["p95_latency"] for r in results]
        max_latencies = [r["max_latency"] for r in results]

        overall_avg = statistics.mean(avg_latencies)
        overall_p95 = statistics.mean(p95_latencies)
        overall_max = max(max_latencies)

        print(f"\nConcurrent access stress test results:")
        print(f"  Users: {num_concurrent_users}")
        print(f"  Total operations: {num_concurrent_users * interactions_per_user}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average latency: {overall_avg * 1000:.1f}ms")
        print(f"  P95 latency: {overall_p95 * 1000:.1f}ms")
        print(f"  Max latency: {overall_max * 1000:.1f}ms")

        # Performance assertions
        assert overall_avg < 0.1, f"Average latency {overall_avg * 1000:.1f}ms exceeds 100ms target"
        assert overall_p95 < 0.5, f"P95 latency {overall_p95 * 1000:.1f}ms exceeds 500ms target"

        # Verify data integrity
        stats = await memory.get_stats()
        assert (
            stats["total_interactions"] >= num_concurrent_users * interactions_per_user * 0.25
        )  # At least 25% were writes

    @pytest.mark.asyncio
    async def test_memory_growth_limits(self, stress_test_system):
        """Test memory growth and cleanup behavior."""
        memory, _, _, interaction_logger, _, hybrid_store = stress_test_system

        # Track memory usage over time
        memory_snapshots = []

        # Phase 1: Rapid growth
        for batch in range(10):
            # Add 1000 interactions per batch
            for i in range(1000):
                # Use new interaction API
                interaction_id = interaction_logger.start_interaction(
                    user_query=f"Batch {batch} operation {i}"
                )
                interaction_logger.log_action(
                    action_type=f"action_{batch}_{i}",
                    target=f"file_{i % 50}.py",
                    details={"batch": batch, "data": "x" * 100},  # Some payload
                )
                completed = interaction_logger.complete_interaction(success=True)
                if completed:
                    await hybrid_store.add_interaction(completed)

            # Check memory stats
            stats = await memory.get_stats()
            memory_snapshots.append(
                {
                    "batch": batch,
                    "interactions": stats["total_interactions"],
                    "patterns": stats["total_patterns"],
                }
            )

        # Phase 2: Consolidation
        consolidation_result = await memory.consolidate_memories()

        # Phase 3: Continued growth with old data
        base_time = datetime.now() - timedelta(days=30)

        # Add old interactions
        for i in range(1000):
            timestamp = base_time + timedelta(minutes=i)
            await memory.store.add_interaction(
                {
                    "session_id": "old_session",
                    "action": f"old_action_{i}",
                    "file_path": f"old_file_{i}.py",
                    "details": {},
                    "timestamp": timestamp.isoformat(),
                }
            )

        # Run consolidation again
        consolidation_result2 = await memory.consolidate_memories()

        # Verify consolidation is working
        assert (
            consolidation_result["episodic_to_semantic"] > 0
            or consolidation_result2["episodic_to_semantic"] > 0
        ), "Should consolidate some memories to semantic level"

        # Check final stats
        final_stats = await memory.get_stats()

        print(f"\nMemory growth test results:")
        print(f"  Total interactions: {final_stats['total_interactions']}")
        print(f"  Total patterns: {final_stats['total_patterns']}")
        print(f"  Consolidation stats: {consolidation_result}")
        print(f"  Old data eligible for cleanup: {consolidation_result2.get('cleaned_up', 0)}")

    @pytest.mark.asyncio
    async def test_pattern_detection_scaling(self, stress_test_system):
        """Test pattern detection performance as data scales."""
        (
            memory,
            pattern_detector,
            _,
            interaction_logger,
            interaction_pattern_detector,
            hybrid_store,
        ) = stress_test_system

        pattern_timings = []
        data_sizes = [100, 500, 1000, 5000]

        for size in data_sizes:
            # Clear previous data
            await memory.clear_specific("all")

            # Generate interactions
            base_time = datetime.now()
            common_patterns = [
                ["edit", "save", "test"],
                ["open", "read", "close"],
                ["checkout", "merge", "push"],
            ]

            # Create interactions with patterns
            interactions = []
            for i in range(size):
                pattern = random.choice(common_patterns)
                # Create an interaction for the pattern
                interaction_id = interaction_logger.start_interaction(
                    user_query=f"Pattern test {i}"
                )

                for j, action in enumerate(pattern):
                    interaction_logger.log_action(action_type=action, target=f"file_{i % 10}.py")

                completed = interaction_logger.complete_interaction(success=True)
                if completed:
                    interactions.append(completed)
                    await hybrid_store.add_interaction(completed)

            # Time pattern detection with new detector
            start_time = time.time()
            patterns = await interaction_pattern_detector.detect_patterns(
                interactions[:100]
            )  # Use subset for performance
            detection_time = time.time() - start_time

            pattern_timings.append(
                {"size": size, "time": detection_time, "patterns_found": len(patterns)}
            )

            print(f"\n  {size} interactions: {detection_time:.3f}s, {len(patterns)} patterns found")

        # Check scaling behavior (should be roughly linear or better)
        time_ratios = []
        for i in range(1, len(pattern_timings)):
            size_ratio = pattern_timings[i]["size"] / pattern_timings[i - 1]["size"]
            time_ratio = pattern_timings[i]["time"] / pattern_timings[i - 1]["time"]
            time_ratios.append(time_ratio / size_ratio)

        avg_scaling_factor = statistics.mean(time_ratios)

        print(f"\nPattern detection scaling factor: {avg_scaling_factor:.2f}")
        assert avg_scaling_factor < 1.5, (
            f"Pattern detection scaling poorly: {avg_scaling_factor:.2f}x worse than linear"
        )

    @pytest.mark.asyncio
    async def test_search_performance(self, stress_test_system):
        """Test search performance with various query types."""
        memory, _, _, interaction_logger, _, hybrid_store = stress_test_system

        # Populate with diverse data
        languages = ["python", "javascript", "go", "rust", "java"]
        operations = ["create", "update", "delete", "read", "validate"]

        for i in range(5000):
            lang = random.choice(languages)
            op = random.choice(operations)

            # Use new interaction API
            interaction_id = interaction_logger.start_interaction(user_query=f"{op} {lang} code")
            interaction_logger.log_action(
                action_type=f"{op}_{lang}_code",
                target=f"src/{lang}/module_{i % 100}.{lang[:2]}",
                details={
                    "language": lang,
                    "operation": op,
                    "complexity": random.choice(["low", "medium", "high"]),
                },
            )
            completed = interaction_logger.complete_interaction(success=True)
            if completed:
                await hybrid_store.add_interaction(completed)

        # Add some patterns
        for i in range(20):
            await memory.store.add_pattern(
                {
                    "name": f"Pattern {i}: {random.choice(operations)} workflow",
                    "description": f"Common {random.choice(languages)} pattern",
                    "pattern_type": "workflow",
                    "sequence": random.sample(operations, 3),
                    "frequency": random.randint(5, 50),
                    "confidence": random.uniform(0.6, 0.95),
                }
            )

        # Test different search scenarios
        search_scenarios = [
            ("Simple keyword", "python"),
            ("File path", "module_42"),
            ("Complex query", "create python high"),
            ("Pattern search", "workflow"),
            ("Non-existent", "xamarin"),
            ("Short query", "go"),
            ("Long query", " ".join(random.choices(string.ascii_lowercase, k=50))),
        ]

        search_results = []

        for scenario_name, query in search_scenarios:
            start_time = time.time()
            results = await memory.search(query)
            search_time = time.time() - start_time

            search_results.append(
                {
                    "scenario": scenario_name,
                    "query": query[:20] + "..." if len(query) > 20 else query,
                    "time": search_time,
                    "results": len(results),
                }
            )

            # All searches should complete quickly
            assert search_time < 0.1, (
                f"Search '{scenario_name}' took {search_time * 1000:.1f}ms, target < 100ms"
            )

        print("\nSearch performance results:")
        for result in search_results:
            print(
                f"  {result['scenario']}: {result['time'] * 1000:.1f}ms, {result['results']} results"
            )

        # Calculate average search time
        avg_search_time = statistics.mean(r["time"] for r in search_results)
        assert avg_search_time < 0.05, (
            f"Average search time {avg_search_time * 1000:.1f}ms exceeds 50ms target"
        )
