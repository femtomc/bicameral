"""Memory consolidation using LLM-based world models."""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_logger
from .llm_service import LLMService
from .world_model import WorldModelInferencer, WorldState

logger = get_logger("memory_consolidator")


class MemoryConsolidator:
    """Manages memory consolidation using dynamic LLM-based understanding."""

    def __init__(self, memory, llm_service: LLMService):
        self.memory = memory
        self.llm_service = llm_service

        # Count-based thresholds for memory transitions
        self.active_to_working_threshold = 10  # Consolidate after 10 interactions
        self.working_to_episodic_threshold = 5  # Consolidate after 5 working memories
        self.episodic_to_semantic_threshold = 10  # Extract semantic after 10 episodes

        # Consolidation parameters
        self.min_frequency_for_semantic = 5  # Min occurrences to become semantic
        self.similarity_threshold = 0.8  # For grouping similar memories
        self.min_interactions_per_session = 3  # Min interactions to form a work session

        # World model components
        self.world_inferencer = WorldModelInferencer(
            llm_service, store=memory.store if memory else None
        )

    async def consolidate_memories(self) -> Dict[str, int]:
        """Run memory consolidation process based on interaction counts."""
        stats = {
            "active_to_working": 0,
            "working_to_episodic": 0,
            "episodic_to_semantic": 0,
            "world_models_updated": 0,
            "interactions_archived": 0,
            "patterns_promoted": 0,
        }

        logger.info("Starting memory consolidation cycle")

        try:
            # STEP 1: Active → Working Memory Consolidation
            # Get unconsolidated interactions
            unconsolidated = await self._get_unconsolidated_interactions()

            if len(unconsolidated) >= self.active_to_working_threshold:
                # Use LLM to identify work sessions and patterns
                sessions = await self._identify_work_sessions_with_llm(unconsolidated)

                for session in sessions:
                    if len(session["interactions"]) >= self.min_interactions_per_session:
                        working_memory = await self._create_working_memory_with_llm(session)
                        if working_memory:
                            await self.memory.store.add_pattern(working_memory)
                            stats["active_to_working"] += 1

                            # Mark interactions as consolidated
                            for interaction in session["interactions"]:
                                interaction["consolidated"] = True
                                interaction["consolidated_to"] = working_memory["id"]

            # STEP 2: Working → Episodic Memory Consolidation
            working_memories = await self.memory.store.get_patterns(
                pattern_type="consolidated_working"
            )

            if len(working_memories) >= self.working_to_episodic_threshold:
                # Use LLM to group related working memories
                episodes = await self._create_episodic_memories_with_llm(working_memories)

                for episode in episodes:
                    await self.memory.store.add_pattern(episode)
                    stats["working_to_episodic"] += 1

            # STEP 3: Episodic → Semantic Knowledge Extraction
            episodic_memories = await self.memory.store.get_patterns(pattern_type="episodic_memory")

            if len(episodic_memories) >= self.episodic_to_semantic_threshold:
                # Use LLM to extract semantic knowledge
                semantic_knowledge = await self._extract_semantic_knowledge_with_llm(
                    episodic_memories
                )

                for knowledge in semantic_knowledge:
                    await self.memory.store.add_pattern(knowledge)
                    stats["episodic_to_semantic"] += 1

            # STEP 4: Build/Update Dynamic World Models
            if unconsolidated:
                world_model = await self._consolidate_to_world_model(unconsolidated)
                if world_model:
                    stats["world_models_updated"] = 1

                    # Get insights from world model
                    insights = self.world_inferencer.get_insights()
                    logger.info(
                        "World model insights",
                        extra={
                            "domain": insights["domain"],
                            "entity_types": insights["discovered_entity_types"],
                            "relation_types": insights["discovered_relation_types"],
                            "total_entities": insights["total_entities"],
                        },
                    )

            # STEP 5: Cleanup old memories
            archived = await self._cleanup_promoted_memories()
            stats["interactions_archived"] = archived

            logger.info("Memory consolidation completed", extra={"stats": stats})

        except Exception as e:
            logger.error("Memory consolidation failed", extra={"error": str(e)}, exc_info=True)

        return stats

    async def _get_unconsolidated_interactions(self) -> List[Dict[str, Any]]:
        """Get interactions that haven't been consolidated yet."""
        # Get recent complete interactions
        all_interactions = await self.memory.store.get_complete_interactions(limit=100)

        # Filter out already consolidated ones
        unconsolidated = []
        for interaction in all_interactions:
            data = interaction if isinstance(interaction, dict) else json.loads(interaction)
            if not data.get("consolidated", False):
                unconsolidated.append(data)

        return unconsolidated

    async def _identify_work_sessions_with_llm(
        self, interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to identify natural work sessions from interactions."""
        # Build prompt for session identification
        interaction_summaries = []
        for i, interaction in enumerate(interactions):
            interaction_summaries.append(
                {
                    "index": i,
                    "timestamp": interaction.get("timestamp"),
                    "query": interaction.get("user_query", "")[:100],
                    "success": interaction.get("success", False),
                }
            )

        prompt = f"""Analyze these interactions and group them into natural work sessions.
A work session is a sequence of related interactions working toward a common goal.

Interactions:
{json.dumps(interaction_summaries, indent=2)}

Group the interactions by their index numbers into sessions. Return JSON:
{{
    "sessions": [
        {{
            "indices": [0, 1, 2],
            "goal": "Brief description of session goal",
            "confidence": 0.8
        }}
    ]
}}"""

        response = await self.llm_service._execute_request(
            self.llm_service._build_request(prompt=prompt, response_format="json", temperature=0.3)
        )

        if response.error or not response.content:
            # Fallback to time-based grouping
            return self._fallback_session_grouping(interactions)

        # Build sessions from LLM response
        sessions = []
        for session_data in response.content.get("sessions", []):
            session_interactions = [
                interactions[i] for i in session_data["indices"] if i < len(interactions)
            ]

            if session_interactions:
                sessions.append(
                    {
                        "interactions": session_interactions,
                        "goal": session_data.get("goal", "Unknown goal"),
                        "confidence": session_data.get("confidence", 0.5),
                    }
                )

        return sessions

    async def _create_working_memory_with_llm(
        self, session: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a working memory entry from a session using LLM analysis."""
        # Build prompt for pattern extraction
        actions_summary = []
        for interaction in session["interactions"]:
            actions_summary.extend(interaction.get("actions_taken", []))

        prompt = f"""Analyze this work session and extract the key pattern or workflow.

Session Goal: {session["goal"]}
Number of Interactions: {len(session["interactions"])}
Actions Taken: {json.dumps(actions_summary, indent=2)}

Extract:
1. The core workflow or pattern
2. Key insights about user preferences
3. Recommendations for future similar tasks

Return JSON with pattern details."""

        response = await self.llm_service._execute_request(
            self.llm_service._build_request(prompt=prompt, response_format="json", temperature=0.5)
        )

        if response.error:
            return None

        pattern_data = response.content

        # Create working memory pattern
        pattern = {
            "name": f"Workflow: {session['goal']}",
            "description": pattern_data.get("workflow", "Extracted workflow pattern"),
            "pattern_type": "consolidated_working",
            "sequence": {
                "actions": actions_summary,
                "insights": pattern_data.get("insights", []),
                "recommendations": pattern_data.get("recommendations", []),
            },
            "frequency": len(session["interactions"]),
            "confidence": session["confidence"],
            "metadata": {
                "session_goal": session["goal"],
                "interaction_count": len(session["interactions"]),
                "llm_extracted": True,
            },
        }

        return pattern

    async def _create_episodic_memories_with_llm(
        self, working_memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to create episodic memories from working memories."""
        # Group similar working memories
        memory_summaries = []
        for mem in working_memories[:20]:  # Limit to recent 20
            memory_summaries.append(
                {
                    "name": mem["name"],
                    "frequency": mem.get("frequency", 1),
                    "confidence": mem.get("confidence", 0.5),
                }
            )

        prompt = f"""Analyze these working memory patterns and identify broader episodes or themes.

Working Memories:
{json.dumps(memory_summaries, indent=2)}

Group related patterns into episodes that represent complete workflows or objectives.
Return JSON with episode descriptions."""

        response = await self.llm_service._execute_request(
            self.llm_service._build_request(prompt=prompt, response_format="json", temperature=0.5)
        )

        if response.error:
            return []

        episodes = []
        for episode_data in response.content.get("episodes", []):
            episode = {
                "name": f"Episode: {episode_data.get('theme', 'Unknown')}",
                "description": episode_data.get("description", ""),
                "pattern_type": "episodic_memory",
                "sequence": episode_data,
                "frequency": episode_data.get("frequency", 1),
                "confidence": episode_data.get("confidence", 0.7),
                "metadata": {"llm_generated": True, "source_count": len(working_memories)},
            }
            episodes.append(episode)

        return episodes

    async def _extract_semantic_knowledge_with_llm(
        self, episodic_memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to extract semantic knowledge from episodic memories."""
        # Summarize episodes
        episode_summaries = []
        for ep in episodic_memories[:15]:  # Limit to recent 15
            episode_summaries.append(
                {
                    "name": ep["name"],
                    "description": ep.get("description", ""),
                    "frequency": ep.get("frequency", 1),
                }
            )

        prompt = f"""Analyze these episodic memories and extract high-level semantic knowledge.

Episodes:
{json.dumps(episode_summaries, indent=2)}

Extract:
1. Core principles or patterns that apply across episodes
2. User preferences and working style
3. Domain-specific insights
4. Recommendations for optimizing workflows

Return JSON with semantic knowledge entries."""

        response = await self.llm_service._execute_request(
            self.llm_service._build_request(prompt=prompt, response_format="json", temperature=0.6)
        )

        if response.error:
            return []

        knowledge_entries = []
        for knowledge_data in response.content.get("knowledge", []):
            entry = {
                "name": f"Principle: {knowledge_data.get('principle', 'Unknown')}",
                "description": knowledge_data.get("description", ""),
                "pattern_type": "semantic_knowledge",
                "sequence": knowledge_data,
                "frequency": len(episodic_memories),
                "confidence": knowledge_data.get("confidence", 0.8),
                "metadata": {
                    "knowledge_type": knowledge_data.get("type", "general"),
                    "llm_extracted": True,
                    "applicability": knowledge_data.get("applicability", "general"),
                },
            }
            knowledge_entries.append(entry)

        return knowledge_entries

    async def _consolidate_to_world_model(
        self, interactions: List[Dict[str, Any]]
    ) -> Optional[WorldState]:
        """Build world model from interactions using LLM."""
        try:
            # Process each interaction through the world model
            for interaction in interactions:
                await self.world_inferencer.infer_from_interaction(interaction)

            # Get the current world state
            world_state = self.world_inferencer.current_world

            # Store as semantic memory
            await self._store_world_as_semantic(world_state)

            return world_state

        except Exception as e:
            logger.error(f"Failed to consolidate world model: {e}")
            return None

    async def _store_world_as_semantic(self, world: WorldState):
        """Store world model as semantic knowledge."""
        semantic_entry = {
            "name": f"World Model: {world.domain or 'Emerging'}",
            "description": f"Dynamic world model with {len(world.entities)} entities",
            "confidence": world.domain_confidence,
            "pattern_type": "semantic_knowledge",
            "sequence": world.to_dict(),
            "frequency": len(world.entities),
            "metadata": {
                "domain": world.domain,
                "entity_types": list(world.discovered_entity_types),
                "relation_types": list(world.discovered_relation_types),
                "goal_count": len(world.inferred_goals),
                "is_world_model": True,
                "dynamic": True,
            },
        }

        await self.memory.store.add_pattern(semantic_entry)

    def _fallback_session_grouping(
        self, interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback to time-based session grouping."""
        sessions = []
        current_session = []

        for i, interaction in enumerate(interactions):
            if not current_session:
                current_session.append(interaction)
            else:
                # Check time gap
                last_time = datetime.fromisoformat(current_session[-1]["timestamp"])
                curr_time = datetime.fromisoformat(interaction["timestamp"])

                if curr_time - last_time < timedelta(minutes=30):
                    current_session.append(interaction)
                else:
                    # Start new session
                    if len(current_session) >= self.min_interactions_per_session:
                        sessions.append(
                            {
                                "interactions": current_session,
                                "goal": "Time-based session",
                                "confidence": 0.5,
                            }
                        )
                    current_session = [interaction]

        # Don't forget the last session
        if len(current_session) >= self.min_interactions_per_session:
            sessions.append(
                {"interactions": current_session, "goal": "Time-based session", "confidence": 0.5}
            )

        return sessions

    async def _cleanup_promoted_memories(self) -> int:
        """Archive consolidated memories."""
        # This would archive old interactions that have been consolidated
        # Implementation depends on your archive strategy
        return 0
