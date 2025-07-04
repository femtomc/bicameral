"""Dynamic world model using LLM inference instead of hardcoded patterns.

This module provides a flexible, learning-based approach to understanding
user interactions and building world models without any domain assumptions.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils.logging_config import get_logger
from .llm_service import LLMService

logger = get_logger("world_model")


@dataclass
class Entity:
    """A dynamically discovered entity in the world model."""

    id: str
    type: str  # No longer an enum - can be anything!
    properties: Dict[str, Any] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5

    def update_property(self, key: str, value: Any):
        """Update a property and track modification time."""
        self.properties[key] = value
        self.last_modified = datetime.now()


@dataclass
class Relation:
    """A dynamically discovered relationship between entities."""

    source_id: str
    target_id: str
    type: str  # No longer an enum - can be anything!
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    observed_count: int = 1


@dataclass
class WorldState:
    """Represents the current understanding of the world without assumptions."""

    domain: Optional[str] = None
    domain_confidence: float = 0.0
    _raw_llm_response: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    inferred_goals: List[Dict[str, Any]] = field(default_factory=list)

    # Track discovered entity and relation types
    discovered_entity_types: Set[str] = field(default_factory=set)
    discovered_relation_types: Set[str] = field(default_factory=set)

    # Persistence metadata
    id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None
    interaction_count: int = 0

    def add_entity(self, entity_id: str, entity_type: str, **properties) -> Entity:
        """Add or update an entity in the world."""
        # Track new entity types
        self.discovered_entity_types.add(entity_type)

        if entity_id in self.entities:
            entity = self.entities[entity_id]
            for key, value in properties.items():
                entity.update_property(key, value)
        else:
            entity = Entity(id=entity_id, type=entity_type, properties=properties)
            self.entities[entity_id] = entity
        return entity

    def add_relation(
        self, source_id: str, target_id: str, relation_type: str, **properties
    ) -> Relation:
        """Add a relationship between entities."""
        # Track new relation types
        self.discovered_relation_types.add(relation_type)

        # Check if relation already exists
        for rel in self.relations:
            if (
                rel.source_id == source_id
                and rel.target_id == target_id
                and rel.type == relation_type
            ):
                rel.observed_count += 1
                rel.confidence = min(0.95, rel.confidence + 0.1)
                return rel

        # Create new relation
        relation = Relation(
            source_id=source_id, target_id=target_id, type=relation_type, properties=properties
        )
        self.relations.append(relation)
        return relation

    def get_entity_relations(self, entity_id: str) -> List[Relation]:
        """Get all relations involving an entity."""
        return [
            rel
            for rel in self.relations
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert world state to dictionary for persistence."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "domain": self.domain,
            "domain_confidence": self.domain_confidence,
            "entities": self.entities,  # Will be serialized by SQLiteStore
            "relations": self.relations,
            "metrics": self.metrics,
            "constraints": self.constraints,
            "inferred_goals": self.inferred_goals,
            "discovered_entity_types": self.discovered_entity_types,
            "discovered_relation_types": self.discovered_relation_types,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "interaction_count": self.interaction_count,
            "_raw_llm_response": self._raw_llm_response,
        }


class WorldModelInferencer:
    """Infers world models from interactions using LLM intelligence."""

    def __init__(self, llm_service: LLMService, store=None):
        self.llm_service = llm_service
        self.store = store  # SQLiteStore for persistence
        self.current_world = WorldState(id=str(uuid.uuid4()), created_at=datetime.now())
        self.interaction_history = []

    async def infer_from_interaction(self, interaction: Dict[str, Any]) -> WorldState:
        """Infer world state updates from a complete interaction using LLM."""
        # Store interaction for context
        self.interaction_history.append(interaction)

        # Use LLM to infer world model updates
        llm_response = await self.llm_service.infer_world_model(interaction)

        if llm_response.error:
            logger.error(f"LLM inference failed: {llm_response.error}")
            # Fall back to basic inference
            return self._basic_inference(interaction)

        # Process LLM response
        inferred_data = llm_response.content

        # Store raw response for debugging
        self.current_world._raw_llm_response = inferred_data

        # Update domain understanding
        if inferred_data.get("domain"):
            self.current_world.domain = inferred_data["domain"]
            self.current_world.domain_confidence = inferred_data.get("confidence", 0.5)

        # Process discovered entities
        for entity_data in inferred_data.get("entities", []):
            self.current_world.add_entity(
                entity_id=entity_data["id"],
                entity_type=entity_data["type"],
                **entity_data.get("properties", {}),
            )

        # Process discovered relations
        for relation_data in inferred_data.get("relations", []):
            self.current_world.add_relation(
                source_id=relation_data["source"],
                target_id=relation_data["target"],
                relation_type=relation_data["type"],
                **relation_data.get("properties", {}),
            )

        # Update goals
        for goal in inferred_data.get("goals", []):
            # Enhance goal with interaction outcome
            goal["achieved"] = interaction.get("success", False)
            goal["timestamp"] = datetime.now().isoformat()
            self.current_world.inferred_goals.append(goal)

        # Update metrics
        self._update_metrics(interaction)

        # Update persistence metadata
        self.current_world.interaction_count += 1
        if not self.current_world.session_id and interaction.get("session_id"):
            self.current_world.session_id = interaction["session_id"]

        # Persist to storage if available
        if self.store:
            try:
                await self.store.add_world_model_state(self.current_world.to_dict())
                logger.info(f"Persisted world model state {self.current_world.id}")
            except Exception as e:
                logger.error(f"Failed to persist world model: {e}")

        logger.info(
            "World model updated",
            extra={
                "domain": self.current_world.domain,
                "entities": len(self.current_world.entities),
                "relations": len(self.current_world.relations),
                "goals": len(self.current_world.inferred_goals),
                "entity_types": list(self.current_world.discovered_entity_types),
                "relation_types": list(self.current_world.discovered_relation_types),
            },
        )

        return self.current_world

    def _basic_inference(self, interaction: Dict[str, Any]) -> WorldState:
        """Basic inference without LLM (fallback)."""
        # Extract any files or targets from actions
        for action in interaction.get("actions_taken", []):
            target = action.get("file") or action.get("target")
            if target:
                # Add as entity with generic type
                self.current_world.add_entity(
                    entity_id=target,
                    entity_type="file" if "." in target else "target",
                    last_action=action.get("action", "unknown"),
                )

        # Basic goal inference from query
        if interaction.get("user_query"):
            goal = {
                "type": "user_intent",
                "description": interaction["user_query"][:100],
                "achieved": interaction.get("success", False),
                "timestamp": datetime.now().isoformat(),
            }
            self.current_world.inferred_goals.append(goal)

        self._update_metrics(interaction)

        # Update persistence metadata
        self.current_world.interaction_count += 1
        if not self.current_world.session_id and interaction.get("session_id"):
            self.current_world.session_id = interaction["session_id"]

        # Note: Persistence would need to be handled by the async caller
        # since this is a sync fallback method

        return self.current_world

    def _update_metrics(self, interaction: Dict[str, Any]):
        """Update world metrics from interaction."""
        # Track success rate
        if "success_rate" not in self.current_world.metrics:
            self.current_world.metrics["success_rate"] = []

        self.current_world.metrics["success_rate"].append(
            1.0 if interaction.get("success") else 0.0
        )

        # Track interaction count
        self.current_world.metrics["interaction_count"] = (
            self.current_world.metrics.get("interaction_count", 0) + 1
        )

        # Track discovered diversity
        self.current_world.metrics["entity_type_diversity"] = len(
            self.current_world.discovered_entity_types
        )
        self.current_world.metrics["relation_type_diversity"] = len(
            self.current_world.discovered_relation_types
        )

    def get_insights(self) -> Dict[str, Any]:
        """Get insights about the discovered world model."""
        return {
            "domain": self.current_world.domain,
            "domain_confidence": self.current_world.domain_confidence,
            "total_entities": len(self.current_world.entities),
            "total_relations": len(self.current_world.relations),
            "discovered_entity_types": list(self.current_world.discovered_entity_types),
            "discovered_relation_types": list(self.current_world.discovered_relation_types),
            "goal_achievement_rate": self._calculate_goal_achievement_rate(),
            "most_connected_entities": self._find_most_connected_entities(),
            "interaction_count": self.current_world.metrics.get("interaction_count", 0),
        }

    def _calculate_goal_achievement_rate(self) -> float:
        """Calculate the rate of achieved goals."""
        if not self.current_world.inferred_goals:
            return 0.0

        achieved = sum(1 for g in self.current_world.inferred_goals if g.get("achieved"))
        return achieved / len(self.current_world.inferred_goals)

    def _find_most_connected_entities(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Find entities with the most relationships."""
        connection_counts = defaultdict(int)

        for relation in self.current_world.relations:
            connection_counts[relation.source_id] += 1
            connection_counts[relation.target_id] += 1

        # Sort by connection count
        sorted_entities = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)

        return sorted_entities[:limit]
