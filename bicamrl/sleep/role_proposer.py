"""Role proposer that discovers behavioral mindsets through world model analysis.

This module implements role proposal as an LLM program that:
1. Inspects world models and memories
2. Identifies behavioral patterns
3. Creates optimized prompts for specific mindsets
4. Uses culturally diverse naming (names from around the world)
"""

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.llm_service import LLMService
from ..core.memory import Memory
from ..core.world_model import WorldModelInferencer
from ..storage.hybrid_store import HybridStore
from ..utils.logging_config import get_logger
from .roles import CommandRole, CommunicationProfile, ContextTrigger, DecisionRule

logger = get_logger("role_proposer")


# Names from various cultures around the world for role naming
CULTURAL_NAMES = {
    "japanese": ["Akira", "Yuki", "Haru", "Kaze", "Sora", "Ren", "Kai", "Mizu"],
    "swahili": ["Amani", "Jabari", "Nia", "Zuberi", "Imara", "Jengo", "Baraka", "Dalili"],
    "arabic": ["Zahra", "Rashid", "Layla", "Hakim", "Nour", "Samir", "Yasmin", "Tariq"],
    "hindi": ["Arjun", "Priya", "Ravi", "Ananya", "Vikram", "Kavya", "Dhruv", "Ishaan"],
    "spanish": ["Luna", "Diego", "Sofia", "Mateo", "Carmen", "Javier", "Elena", "Pablo"],
    "yoruba": ["Adunni", "Ayodele", "Folake", "Kayode", "Omolara", "Temitope", "Yemi", "Bayo"],
    "russian": ["Anastasia", "Dmitri", "Svetlana", "Alexei", "Natasha", "Mikhail", "Olga", "Ivan"],
    "korean": [
        "Min-jun",
        "Seo-yeon",
        "Ha-joon",
        "Ji-woo",
        "Sung-ho",
        "Eun-ji",
        "Tae-hyung",
        "Yuna",
    ],
    "quechua": ["Inti", "Killa", "Pacha", "Ayar", "Mama", "Waman", "Qoya", "Kuntur"],
    "maori": ["Aroha", "Koa", "Manaia", "Tane", "Hine", "Rangi", "Moana", "Atua"],
    "portuguese": ["João", "Ana", "Pedro", "Maria", "Carlos", "Beatriz", "Miguel", "Clara"],
    "bengali": ["Arko", "Diya", "Ayan", "Riya", "Soham", "Tara", "Dev", "Mira"],
    "french": ["Amélie", "Louis", "Céline", "Hugo", "Margot", "Théo", "Camille", "Lucas"],
    "zulu": ["Amahle", "Bongani", "Lerato", "Sibusiso", "Nomusa", "Thabo", "Zanele", "Lwazi"],
    "mandarin": ["Wei", "Mei", "Chen", "Li", "Xiao", "Yu", "Jing", "Hao"],
    "greek": ["Athena", "Dimitris", "Eleni", "Nikos", "Sophia", "Yannis", "Maria", "Alexis"],
}


def generate_role_name() -> str:
    """Generate a random culturally diverse name for a role."""
    culture = random.choice(list(CULTURAL_NAMES.keys()))
    name = random.choice(CULTURAL_NAMES[culture])
    return name


@dataclass
class BehavioralPattern:
    """A discovered behavioral pattern from world model analysis."""

    pattern_id: str
    domain: str
    recurring_goals: List[str]
    key_entities: List[str]
    action_sequences: List[List[str]]
    success_indicators: Dict[str, Any]
    context_conditions: Dict[str, Any]
    frequency: int
    confidence: float


class RoleProposer:
    """Discovers roles by analyzing world models and memories as an LLM program.

    This is an LLM agent that:
    1. Queries world models to understand domains and goals
    2. Analyzes memories to find behavioral patterns
    3. Creates optimized prompts for specific mindsets
    4. Names roles using culturally diverse names
    """

    def __init__(
        self,
        memory: Memory,
        world_model: WorldModelInferencer,
        llm_service: LLMService,
        hybrid_store: Optional[HybridStore] = None,
    ):
        self.memory = memory
        self.world_model = world_model
        self.llm_service = llm_service
        self.hybrid_store = hybrid_store

    async def discover_roles(self, min_interactions: int = 10) -> List[CommandRole]:
        """Main LLM program to discover roles from world models and memories."""

        logger.info("Starting LLM-driven role discovery")

        # Step 1: Query world models to understand domains
        world_insights = await self._analyze_world_models()

        # Step 2: Analyze memories for behavioral patterns
        behavioral_patterns = await self._discover_behavioral_patterns(world_insights)

        # Step 3: Create roles from patterns
        roles = await self._create_roles_from_patterns(behavioral_patterns)

        logger.info(f"Discovered {len(roles)} roles through world model analysis")
        return roles

    async def _analyze_world_models(self) -> Dict[str, Any]:
        """LLM analyzes stored world models to understand domains and goals."""

        # Get recent world model states
        world_states = await self.memory.store.get_world_model_states(limit=50)

        if not world_states:
            logger.info("No world model states found")
            return {}

        # Prepare world model summary for LLM
        prompt = f"""Analyze these world model states to identify:
1. Primary domains the user works in
2. Recurring goals and objectives
3. Key entity types and relationships
4. Patterns in how domains evolve over time

World Model States:
{json.dumps(world_states[:10], indent=2, default=str)}

Provide insights as JSON with structure:
{{
    "domains": [
        {{"name": "...", "frequency": 0.0, "key_activities": [...]}},
        ...
    ],
    "recurring_goals": [
        {{"goal": "...", "domain": "...", "frequency": 0.0}},
        ...
    ],
    "entity_patterns": [
        {{"type": "...", "common_relations": [...], "importance": 0.0}},
        ...
    ],
    "behavioral_insights": [
        "insight1",
        "insight2"
    ]
}}"""

        response = await self.llm_service.infer_world_model({"prompt": prompt})

        if response.error:
            logger.error(f"Failed to analyze world models: {response.error}")
            return {}

        return response.content

    async def _discover_behavioral_patterns(
        self, world_insights: Dict[str, Any]
    ) -> List[BehavioralPattern]:
        """LLM discovers behavioral patterns by analyzing memories with world context."""

        # Get recent interactions
        recent_interactions = await self.memory.get_recent_interactions(limit=100)

        if not recent_interactions:
            return []

        # LLM analyzes interactions in context of world insights

        response = await self.llm_service.analyze_patterns(recent_interactions)

        if response.error:
            logger.error(f"Failed to discover patterns: {response.error}")
            return []

        # Parse patterns
        patterns = []
        try:
            pattern_data = response.content.get("patterns", [])
            for p in pattern_data:
                patterns.append(BehavioralPattern(**p))
        except Exception as e:
            logger.error(f"Failed to parse behavioral patterns: {e}")

        return patterns

    async def _create_roles_from_patterns(
        self, patterns: List[BehavioralPattern]
    ) -> List[CommandRole]:
        """LLM creates optimized role prompts from behavioral patterns."""

        roles = []

        for pattern in patterns:
            # Generate culturally diverse name
            role_name = generate_role_name()

            # LLM creates optimized prompt for this behavioral mindset
            prompt = f"""Create an optimized role definition for this behavioral pattern.

Pattern:
{json.dumps(pattern.__dict__, indent=2, default=str)}

Create a role that will help the AI assistant enter the right mindset for this behavior.
The role should:
1. Prime the assistant for the specific domain and goals
2. Emphasize the successful action patterns
3. Include context triggers that activate this mindset
4. Define communication style that fits this behavior

Return as JSON:
{{
    "description": "A one-line description of this behavioral mindset",
    "mindset_prompt": "The optimized prompt to put the assistant in this mindset",
    "context_triggers": [
        {{"condition": "trigger_type", "value": "trigger_value", "weight": 0.0}},
        ...
    ],
    "communication_style": {{
        "tone": "professional/casual/technical",
        "formality": 0.0,
        "verbosity": 0.0,
        "technical_depth": 0.0
    }},
    "tool_preferences": {{"tool_name": weight, ...}},
    "decision_rules": [
        {{"condition": "...", "action": "...", "confidence": 0.0}},
        ...
    ]
}}"""

            response = await self.llm_service.enhance_prompt(prompt, {"pattern": pattern})

            if response.error:
                logger.error(
                    f"Failed to create role for pattern {pattern.pattern_id}: {response.error}"
                )
                continue

            try:
                role_data = json.loads(response.content)

                # Create role with generated name
                role = CommandRole(
                    name=role_name,
                    description=role_data["description"],
                    mindset_prompt=role_data["mindset_prompt"],
                    context_triggers=[ContextTrigger(**t) for t in role_data["context_triggers"]],
                    tool_preferences=role_data["tool_preferences"],
                    decision_rules=[DecisionRule(**r) for r in role_data["decision_rules"]],
                    communication_profile=CommunicationProfile(**role_data["communication_style"]),
                    performance_metrics={
                        "activations": 0,
                        "success_rate": 0.0,
                        "avg_interaction_time": 0.0,
                    },
                    domain=pattern.domain,
                    discovered_from_pattern=pattern.pattern_id,
                )

                roles.append(role)
                logger.info(f"Created role '{role_name}' for {pattern.domain} domain")

            except Exception as e:
                logger.error(f"Failed to parse role data: {e}")

        return roles

    async def refine_role(self, role: CommandRole, feedback: Dict[str, Any]) -> CommandRole:
        """LLM refines a role based on performance feedback."""

        prompt = f"""Refine this role based on performance feedback.

Current Role:
Name: {role.name}
Description: {role.description}
Mindset Prompt: {role.mindset_prompt}
Domain: {role.domain}

Performance Feedback:
{json.dumps(feedback, indent=2)}

Analyze the feedback and suggest improvements to:
1. The mindset prompt to better capture the behavioral pattern
2. Context triggers to be more accurate
3. Communication style adjustments
4. Tool preferences based on actual usage

Return refined role definition in same JSON format as before."""

        response = await self.llm_service.enhance_prompt(
            prompt, {"role": role, "feedback": feedback}
        )

        if response.error:
            logger.error(f"Failed to refine role {role.name}: {response.error}")
            return role

        try:
            refined_data = json.loads(response.content)

            # Update role with refinements
            role.description = refined_data.get("description", role.description)
            role.mindset_prompt = refined_data.get("mindset_prompt", role.mindset_prompt)

            if "context_triggers" in refined_data:
                role.context_triggers = [
                    ContextTrigger(**t) for t in refined_data["context_triggers"]
                ]

            if "communication_style" in refined_data:
                role.communication_profile = CommunicationProfile(
                    **refined_data["communication_style"]
                )

            if "tool_preferences" in refined_data:
                role.tool_preferences = refined_data["tool_preferences"]

            logger.info(f"Refined role '{role.name}' based on feedback")

        except Exception as e:
            logger.error(f"Failed to apply role refinements: {e}")

        return role
