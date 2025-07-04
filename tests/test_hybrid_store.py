"""Test hybrid storage functionality."""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pytest

from bicamrl.core.interaction_model import (
    Action,
    ActionStatus,
    FeedbackType,
    Interaction,
)
from bicamrl.storage.hybrid_store import HybridStore
from bicamrl.storage.llm_embeddings import LLMEmbeddings


class MockEmbeddings:
    """Mock embeddings for testing without LLM."""

    def __init__(self):
        self.embed_count = 0
        self.dimension = 768  # Match typical LLM embedding dimension

    async def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embeddings based on text."""
        # Simple hash-based embedding for testing
        self.embed_count += 1
        hash_val = hash(text)
        # Create a 384-dimensional embedding (like sentence-transformers)
        np.random.seed(abs(hash_val) % (2**32))
        return np.random.randn(self.dimension).astype(np.float32)

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        embeddings = []
        for text in texts:
            emb = await self.embed(text)
            embeddings.append(emb)
        return np.array(embeddings)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


@pytest.fixture
async def hybrid_store(tmp_path):
    """Create a hybrid store with mock embeddings."""
    store_path = tmp_path / "test_hybrid"
    embeddings = MockEmbeddings()
    store = HybridStore(store_path, embeddings)
    yield store
    # Cleanup
    if store_path.exists():
        shutil.rmtree(store_path)


@pytest.fixture
def sample_interactions():
    """Create sample interactions for testing."""
    interactions = []

    # Successful file edit interaction
    interaction1 = Interaction(
        interaction_id="test_1",
        session_id="session_1",
        user_query="Can you fix the typo in the README file?",
        ai_interpretation="User wants me to fix a typo in README.md",
        planned_actions=["read_file", "edit_file"],
        confidence=0.9
    )

    action1 = Action(
        action_type="read_file",
        target="README.md",
        status=ActionStatus.COMPLETED,
        result="Found typo on line 5"
    )
    action2 = Action(
        action_type="edit_file",
        target="README.md",
        status=ActionStatus.COMPLETED,
        result="Fixed typo: 'teh' -> 'the'"
    )
    interaction1.actions_taken = [action1, action2]
    interaction1.user_feedback = "Thanks, that's perfect!"
    interaction1.feedback_type = FeedbackType.APPROVAL
    interaction1.success = True
    interactions.append(interaction1)

    # Corrected interaction
    interaction2 = Interaction(
        interaction_id="test_2",
        session_id="session_1",
        user_query="Can you update the configuration file?",
        ai_interpretation="User wants me to update config.json",
        planned_actions=["edit_file"],
        confidence=0.7
    )

    action3 = Action(
        action_type="edit_file",
        target="config.json",
        status=ActionStatus.COMPLETED
    )
    interaction2.actions_taken = [action3]
    interaction2.user_feedback = "No, I meant the .env configuration file"
    interaction2.feedback_type = FeedbackType.CORRECTION
    interaction2.success = False
    interactions.append(interaction2)

    # Follow-up interaction
    interaction3 = Interaction(
        interaction_id="test_3",
        session_id="session_1",
        user_query="Update the .env file instead",
        ai_interpretation="User wants me to update .env file",
        planned_actions=["edit_file"],
        confidence=0.95
    )

    action4 = Action(
        action_type="edit_file",
        target=".env",
        status=ActionStatus.COMPLETED
    )
    interaction3.actions_taken = [action4]
    interaction3.user_feedback = "Great, now also update the docker config"
    interaction3.feedback_type = FeedbackType.FOLLOWUP
    interaction3.success = True
    interactions.append(interaction3)

    return interactions


class TestHybridStore:
    """Test hybrid storage functionality."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_interaction(self, hybrid_store, sample_interactions):
        """Test storing and retrieving interactions."""
        interaction = sample_interactions[0]

        # Store interaction
        await hybrid_store.add_interaction(interaction)

        # Retrieve by ID
        retrieved = await hybrid_store.get_interaction_by_id(interaction.interaction_id)

        assert retrieved is not None
        assert retrieved.interaction_id == interaction.interaction_id
        assert retrieved.user_query == interaction.user_query
        assert retrieved.success == interaction.success
        assert len(retrieved.actions_taken) == len(interaction.actions_taken)

    @pytest.mark.asyncio
    async def test_similarity_search(self, hybrid_store, sample_interactions):
        """Test similarity search functionality."""
        # Store all interactions
        for interaction in sample_interactions:
            await hybrid_store.add_interaction(interaction)

        # Search for similar queries
        similar = await hybrid_store.search_similar_queries(
            "How do I fix typos in files?",
            k=3,
            threshold=0.0  # Low threshold for mock embeddings
        )

        assert len(similar) > 0
        # Check result structure
        for emb_id, similarity, metadata in similar:
            assert isinstance(emb_id, str)
            assert isinstance(similarity, float)
            assert "text" in metadata
            assert "type" in metadata

    @pytest.mark.asyncio
    async def test_correction_patterns(self, hybrid_store, sample_interactions):
        """Test finding correction patterns."""
        # Store interactions including corrections
        for interaction in sample_interactions:
            await hybrid_store.add_interaction(interaction)

        # Find correction patterns
        corrections = await hybrid_store.find_correction_patterns(limit=5)

        assert len(corrections) >= 1
        for correction in corrections:
            assert "original_query" in correction
            assert "ai_interpretation" in correction
            assert "correction" in correction
            assert "interaction_id" in correction

    @pytest.mark.asyncio
    async def test_successful_patterns(self, hybrid_store, sample_interactions):
        """Test finding successful patterns."""
        # Store interactions
        for interaction in sample_interactions:
            await hybrid_store.add_interaction(interaction)

        # Find successful edit patterns
        successful = await hybrid_store.find_successful_patterns(
            action_types=["edit_file"],
            limit=5
        )

        assert len(successful) >= 1
        for pattern in successful:
            assert "query" in pattern
            assert "actions" in pattern
            assert "confidence" in pattern

    @pytest.mark.asyncio
    async def test_query_clustering(self, hybrid_store):
        """Test query clustering functionality."""
        # Add similar queries for clustering
        similar_queries = [
            "Can you fix the error in file1.py?",
            "Can you fix the error in file2.py?",
            "Can you fix the error in file3.py?",
            "Update the configuration in settings.json",
            "Update the configuration in config.yaml",
        ]

        for i, query in enumerate(similar_queries):
            interaction = Interaction(
                interaction_id=f"cluster_test_{i}",
                session_id="session_cluster",
                user_query=query,
                success=True
            )
            await hybrid_store.add_interaction(interaction)

        # Perform clustering
        clusters = await hybrid_store.cluster_similar_queries(
            min_cluster_size=2,
            similarity_threshold=0.0  # Low threshold for mock embeddings
        )

        assert len(clusters) >= 1
        for cluster in clusters:
            assert len(cluster) >= 2
            for item in cluster:
                assert "text" in item
                assert "interaction_id" in item

    @pytest.mark.asyncio
    async def test_statistics(self, hybrid_store, sample_interactions):
        """Test statistics retrieval."""
        # Store some interactions
        for interaction in sample_interactions:
            await hybrid_store.add_interaction(interaction)

        # Get statistics
        stats = await hybrid_store.get_statistics()

        assert "total_embeddings" in stats
        assert "embedding_dimension" in stats
        assert "storage_size_mb" in stats
        assert "embedding_types" in stats

        assert stats["total_embeddings"] >= len(sample_interactions)
        assert stats["embedding_dimension"] == 768  # Mock embedding dimension
        assert isinstance(stats["storage_size_mb"], float)

    @pytest.mark.asyncio
    async def test_recent_interactions(self, hybrid_store, sample_interactions):
        """Test retrieving recent interactions."""
        # Store interactions with small delay to ensure different timestamps
        for i, interaction in enumerate(sample_interactions):
            await hybrid_store.add_interaction(interaction)
            await asyncio.sleep(0.01)  # Small delay for timestamp ordering

        # Get recent interactions
        recent = await hybrid_store.get_recent_interactions(limit=2)

        assert len(recent) == 2
        # Should be in reverse chronological order
        # Check if interaction_id is in details or at top level
        if "details" in recent[0]:
            assert recent[0]["details"]["interaction_id"] == "test_3"
            assert recent[1]["details"]["interaction_id"] == "test_2"
        else:
            assert recent[0]["interaction_id"] == "test_3"
            assert recent[1]["interaction_id"] == "test_2"

    @pytest.mark.asyncio
    async def test_action_patterns(self, hybrid_store, sample_interactions):
        """Test finding action patterns."""
        # Store interactions
        for interaction in sample_interactions:
            await hybrid_store.add_interaction(interaction)

        # Find action sequences
        sequences = await hybrid_store.find_action_sequences(min_occurrences=1)

        assert len(sequences) > 0
        for seq in sequences:
            assert "actions" in seq
            assert "count" in seq
            assert "success_rate" in seq
            assert isinstance(seq["actions"], list)

    @pytest.mark.asyncio
    async def test_feedback_analysis(self, hybrid_store, sample_interactions):
        """Test feedback type analysis."""
        # Store interactions
        for interaction in sample_interactions:
            await hybrid_store.add_interaction(interaction)

        # Analyze feedback
        feedback_stats = await hybrid_store.get_feedback_statistics()

        assert "total_feedback" in feedback_stats
        assert "by_type" in feedback_stats
        assert FeedbackType.APPROVAL.value in feedback_stats["by_type"]
        assert FeedbackType.CORRECTION.value in feedback_stats["by_type"]
        assert FeedbackType.FOLLOWUP.value in feedback_stats["by_type"]

    @pytest.mark.asyncio
    async def test_embedding_caching(self, hybrid_store):
        """Test that embeddings are cached properly."""
        # Add the same query multiple times
        query = "Test query for caching"

        for i in range(3):
            interaction = Interaction(
                interaction_id=f"cache_test_{i}",
                session_id="session_cache",
                user_query=query,
                success=True
            )
            await hybrid_store.add_interaction(interaction)

        # Search for the query
        results = await hybrid_store.search_similar_queries(query, k=5)

        # Should find all 3 instances
        matching_results = [r for r in results if r[2]["text"] == query]
        assert len(matching_results) == 3

    @pytest.mark.asyncio
    async def test_error_handling(self, hybrid_store):
        """Test error handling for invalid inputs."""
        # Test retrieving non-existent interaction
        result = await hybrid_store.get_interaction_by_id("non_existent")
        assert result is None

        # Test empty query search
        results = await hybrid_store.search_similar_queries("", k=5)
        assert isinstance(results, list)

        # Test invalid interaction
        with pytest.raises(Exception):
            await hybrid_store.add_interaction(None)


class TestVectorStore:
    """Test vector store component directly."""

    @pytest.mark.asyncio
    async def test_vector_operations(self, tmp_path):
        """Test low-level vector operations."""
        from bicamrl.storage.vector_store import VectorStore

        store = VectorStore(tmp_path / "vectors")

        # Add vectors
        vec1 = np.random.randn(128).astype(np.float32)
        vec2 = np.random.randn(128).astype(np.float32)

        # VectorStore.add_embedding signature: (embedding_id, embedding, metadata)
        await store.add_embedding("id1", vec1, {"text": "test1"})
        await store.add_embedding("id2", vec2, {"text": "test2"})

        # Search
        results = await store.search_similar(vec1, k=2)
        assert len(results) == 2
        assert results[0][0] == "id1"  # First result should be id1
        assert results[0][1] > 0.99  # Very high similarity

        # Get by ID
        retrieved = await store.get_embedding("id1")
        assert retrieved is not None
        assert np.allclose(retrieved[0], vec1)
        assert retrieved[1]["text"] == "test1"

        # Statistics
        stats = store.statistics()
        assert stats["count"] == 2
        assert stats["dimension"] == 128
