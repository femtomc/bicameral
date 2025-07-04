"""Hybrid storage combining SQLite and vector storage for semantic search."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.interaction_model import FeedbackType, Interaction
from ..utils.logging_config import get_logger
from .sqlite_store import SQLiteStore

logger = get_logger("hybrid_store")


class HybridStore:
    """Combines SQLite for structured data with vector storage for embeddings."""

    def __init__(self, storage_path: Path, llm_embeddings, vector_backend: str = "basic"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize sub-stores
        self.sqlite_store = SQLiteStore(self.storage_path / "bicameral.db")

        # Initialize vector store based on backend selection
        vector_path = self.storage_path / "vectors"
        if vector_backend == "lancedb":
            from .vector_store import LanceDBVectorStore

            self.vector_store = LanceDBVectorStore(vector_path)
            logger.info("Using LanceDB vector backend")
        elif vector_backend == "chromadb":
            from .vector_store import ChromaDBVectorStore

            self.vector_store = ChromaDBVectorStore(vector_path)
            logger.info("Using ChromaDB vector backend")
        else:
            from .vector_store import VectorStore

            self.vector_store = VectorStore(vector_path)
            logger.info("Using basic numpy vector backend")

        # LLM embeddings (required)
        if not llm_embeddings:
            raise ValueError("LLM embeddings are required")
        self._llm_embeddings = llm_embeddings

        # Create adapter for LLM embeddings
        from .llm_embeddings import LLMEmbeddingAdapter

        self._embedder = LLMEmbeddingAdapter(llm_embeddings)

    def _get_embedder(self):
        """Get the embedder."""
        return self._embedder

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        embedder = self._get_embedder()
        # Always async LLM embeddings
        embeddings = await embedder.encode([text])
        return embeddings[0]

    async def add_interaction(self, interaction: Interaction):
        """Add interaction to both SQLite and vector store."""
        # Store in SQLite
        interaction_dict = interaction.to_dict()

        # Store in both old and new format for compatibility
        await self.sqlite_store.add_interaction(
            {
                "timestamp": interaction_dict["timestamp"],
                "session_id": interaction_dict["session_id"],
                "action": "interaction",  # Generic action type
                "details": interaction_dict,
            }
        )

        # Store in new complete_interactions table
        await self.sqlite_store.add_complete_interaction(interaction_dict)

        # Generate embeddings for user query
        query_embedding = await self._generate_embedding(interaction.user_query)

        # Store query embedding
        await self.vector_store.add_embedding(
            embedding_id=f"query_{interaction.interaction_id}",
            embedding=query_embedding,
            metadata={
                "interaction_id": interaction.interaction_id,
                "type": "user_query",
                "text": interaction.user_query,
                "timestamp": interaction.timestamp.isoformat(),
                "success": interaction.success,
                "actions": interaction.action_sequence,
            },
        )

        # If there's an AI interpretation, embed that too
        if interaction.ai_interpretation:
            interp_embedding = await self._generate_embedding(interaction.ai_interpretation)
            await self.vector_store.add_embedding(
                embedding_id=f"interp_{interaction.interaction_id}",
                embedding=interp_embedding,
                metadata={
                    "interaction_id": interaction.interaction_id,
                    "type": "ai_interpretation",
                    "text": interaction.ai_interpretation,
                    "success": interaction.success,
                },
            )

        # If there's user feedback, embed that as well
        if interaction.user_feedback:
            feedback_embedding = await self._generate_embedding(interaction.user_feedback)
            await self.vector_store.add_embedding(
                embedding_id=f"feedback_{interaction.interaction_id}",
                embedding=feedback_embedding,
                metadata={
                    "interaction_id": interaction.interaction_id,
                    "type": "user_feedback",
                    "text": interaction.user_feedback,
                    "feedback_type": interaction.feedback_type.value,
                },
            )

        logger.info(f"Stored interaction {interaction.interaction_id} with embeddings")

    async def search_similar_queries(
        self, query: str, k: int = 5, threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find interactions with similar user queries."""
        query_embedding = await self._generate_embedding(query)

        # Search in vector store
        results = await self.vector_store.search_similar(
            query_embedding=query_embedding,
            k=k * 2,
            threshold=threshold,  # Get more to filter
        )

        # Filter for user queries only
        query_results = [(r[0], r[1], r[2]) for r in results if r[2].get("type") == "user_query"]

        return query_results[:k]

    async def search_similar_patterns(
        self, pattern: str, pattern_type: str = "all", k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar patterns across all interaction components."""
        pattern_embedding = await self._generate_embedding(pattern)

        # Search all embeddings
        results = await self.vector_store.search_similar(
            query_embedding=pattern_embedding,
            k=k * 3,  # Get more to filter by type
        )

        # Filter by pattern type if specified
        if pattern_type != "all":
            results = [(r[0], r[1], r[2]) for r in results if r[2].get("type") == pattern_type]

        return results[:k]

    async def get_interaction_by_id(self, interaction_id: str) -> Optional[Interaction]:
        """Retrieve full interaction from SQLite by ID."""
        # First try the new complete_interactions table
        interactions = await self.sqlite_store.get_complete_interactions(limit=1000)

        for interaction_data in interactions:
            if interaction_data.get("interaction_id") == interaction_id:
                return Interaction.from_dict(interaction_data)

        # Fallback to old format
        old_interactions = await self.sqlite_store.get_recent_interactions(limit=1000)

        for interaction_data in old_interactions:
            if interaction_data.get("details", {}).get("interaction_id") == interaction_id:
                return Interaction.from_dict(interaction_data["details"])

        return None

    async def find_correction_patterns(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Find patterns where users corrected the AI."""
        # Get all feedback embeddings
        # Use a random embedding instead of zeros to avoid division by zero
        dimension = 768 if self._llm_embeddings else 384
        dummy_embedding = np.random.randn(dimension)
        all_embeddings = await self.vector_store.search_similar(
            query_embedding=dummy_embedding,
            k=1000,
            threshold=-1.0,  # Get many  # Get all
        )

        # Filter for corrections
        corrections = []
        for emb_id, _, metadata in all_embeddings:
            if (
                metadata.get("type") == "user_feedback"
                and metadata.get("feedback_type") == FeedbackType.CORRECTION.value
            ):
                # Get the original interaction
                interaction_id = metadata["interaction_id"]
                interaction = await self.get_interaction_by_id(interaction_id)

                if interaction:
                    corrections.append(
                        {
                            "interaction_id": interaction_id,
                            "original_query": interaction.user_query,
                            "ai_interpretation": interaction.ai_interpretation,
                            "correction": metadata["text"],
                            "timestamp": interaction.timestamp,
                        }
                    )

        return corrections[:limit]

    async def find_successful_patterns(
        self, action_types: Optional[List[str]] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find patterns from successful interactions."""
        # Get query embeddings from successful interactions
        # Use a random embedding instead of zeros to avoid division by zero
        dimension = 768 if self._llm_embeddings else 384
        dummy_embedding = np.random.randn(dimension)
        all_embeddings = await self.vector_store.search_similar(
            query_embedding=dummy_embedding, k=1000, threshold=-1.0
        )

        successful = []
        for emb_id, _, metadata in all_embeddings:
            if metadata.get("type") == "user_query" and metadata.get("success") == True:
                # Filter by action types if specified
                if action_types:
                    interaction_actions = metadata.get("actions", [])
                    if not any(action in interaction_actions for action in action_types):
                        continue

                successful.append(
                    {
                        "query": metadata["text"],
                        "actions": metadata.get("actions", []),
                        "interaction_id": metadata["interaction_id"],
                        "timestamp": metadata["timestamp"],
                        "confidence": metadata.get("confidence", 0.9),  # Add confidence
                    }
                )

        return successful[:limit]

    async def cluster_similar_queries(
        self, min_cluster_size: int = 3, similarity_threshold: float = 0.7
    ) -> List[List[Dict[str, Any]]]:
        """Group similar queries into clusters."""
        # Get all query embeddings
        all_queries = []
        all_embeddings = []

        # Use a random embedding instead of zeros to avoid division by zero
        dimension = 768 if self._llm_embeddings else 384
        dummy_embedding = np.random.randn(dimension)
        results = await self.vector_store.search_similar(
            query_embedding=dummy_embedding, k=1000, threshold=-1.0
        )

        for emb_id, _, metadata in results:
            if metadata.get("type") == "user_query":
                result = await self.vector_store.get_embedding(emb_id)
                if result is not None:
                    embedding, _ = result
                    all_queries.append(metadata)
                    all_embeddings.append(embedding)

        if len(all_queries) < min_cluster_size:
            return []

        # Simple clustering based on similarity threshold
        clusters = []
        used = set()

        for i, (query1, emb1) in enumerate(zip(all_queries, all_embeddings)):
            if i in used:
                continue

            cluster = [query1]
            used.add(i)

            # Find similar queries
            for j, (query2, emb2) in enumerate(zip(all_queries, all_embeddings)):
                if j <= i or j in used:
                    continue

                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

                if similarity > similarity_threshold:  # Use the parameter
                    cluster.append(query2)
                    used.add(j)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    async def get_recent_interactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent interactions from SQLite store."""
        return await self.sqlite_store.get_recent_interactions(limit=limit)

    async def find_action_sequences(self, min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """Find common action sequences."""
        # Get all interactions
        interactions = await self.sqlite_store.get_recent_interactions(limit=10000)

        # Extract action sequences
        sequences = {}
        for interaction in interactions:
            # Check if interaction has details field (old format)
            if "details" in interaction and isinstance(interaction["details"], dict):
                interaction_data = interaction["details"]
            else:
                interaction_data = interaction

            actions_taken = interaction_data.get("actions_taken", [])
            # Handle case where actions_taken might be JSON string
            if isinstance(actions_taken, str):
                actions_taken = json.loads(actions_taken)  # Let it fail!

            if actions_taken:
                # Create sequence key
                actions = [a["action_type"] for a in actions_taken]
                seq_key = ",".join(actions)

                if seq_key not in sequences:
                    sequences[seq_key] = {"actions": actions, "count": 0, "success_count": 0}

                sequences[seq_key]["count"] += 1
                if interaction_data.get("success"):
                    sequences[seq_key]["success_count"] += 1

        # Filter by min occurrences and calculate success rate
        result = []
        for seq_data in sequences.values():
            if seq_data["count"] >= min_occurrences:
                seq_data["success_rate"] = seq_data["success_count"] / seq_data["count"]
                result.append(seq_data)

        return sorted(result, key=lambda x: x["count"], reverse=True)

    async def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        interactions = await self.sqlite_store.get_recent_interactions(limit=10000)

        stats = {"total_feedback": 0, "by_type": {}}

        for interaction in interactions:
            # Check if interaction has details field (old format)
            if "details" in interaction and isinstance(interaction["details"], dict):
                interaction_data = interaction["details"]
            else:
                interaction_data = interaction

            if (
                interaction_data.get("feedback_type")
                and interaction_data.get("feedback_type") != "none"
            ):
                stats["total_feedback"] += 1
                feedback_type = interaction_data["feedback_type"]
                if feedback_type not in stats["by_type"]:
                    stats["by_type"][feedback_type] = 0
                stats["by_type"][feedback_type] += 1

        return stats

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        vector_stats = self.vector_store.statistics()

        # Count different types of embeddings
        # Use a random embedding instead of zeros to avoid division by zero
        dimension = 768 if self._llm_embeddings else 384
        dummy_embedding = np.random.randn(dimension)
        all_embeddings = await self.vector_store.search_similar(
            query_embedding=dummy_embedding, k=10000, threshold=-1.0
        )

        type_counts = {}
        for _, _, metadata in all_embeddings:
            emb_type = metadata.get("type", "unknown")
            type_counts[emb_type] = type_counts.get(emb_type, 0) + 1

        return {
            "total_embeddings": vector_stats["count"],
            "embedding_dimension": vector_stats["dimension"],
            "storage_size_mb": vector_stats["storage_size_mb"],
            "embedding_types": type_counts,
            "deleted_count": vector_stats["deleted_count"],
        }

    async def migrate_existing_interactions(self):
        """Migrate existing SQLite interactions to include embeddings."""
        logger.info("Starting migration of existing interactions...")

        # Get all interactions from SQLite
        interactions = await self.sqlite_store.get_recent_interactions(limit=10000)

        migrated = 0
        for interaction_data in interactions:
            if "details" in interaction_data and isinstance(interaction_data["details"], dict):
                details = interaction_data["details"]

                # Check if this looks like a full interaction
                if "interaction_id" in details and "user_query" in details:
                    try:
                        interaction = Interaction.from_dict(details)

                        # Check if already migrated
                        existing = await self.vector_store.get_embedding(
                            f"query_{interaction.interaction_id}"
                        )
                        if existing is None:
                            # Add embeddings
                            await self.add_interaction(interaction)
                            migrated += 1

                            if migrated % 10 == 0:
                                logger.info(f"Migrated {migrated} interactions...")
                    except Exception as e:
                        logger.error(f"Failed to migrate interaction: {e}")

        logger.info(f"Migration complete. Migrated {migrated} interactions.")
        return migrated
