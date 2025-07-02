"""LLM-based role discovery using interaction analysis."""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

from ..core.memory import Memory
from ..storage.hybrid_store import HybridStore
from ..utils.logging_config import get_logger
from .roles import (
    CommandRole, ContextTrigger, TriggerType,
    DecisionRule, CommunicationProfile, CommunicationStyle
)
from .llm_providers import BaseLLMProvider


logger = get_logger("llm_role_discoverer")


class LLMRoleDiscoverer:
    """Discovers behavioral roles by having LLM analyze interaction patterns."""
    
    def __init__(
        self, 
        memory: Memory,
        hybrid_store: HybridStore,
        llm_provider: BaseLLMProvider
    ):
        self.memory = memory
        self.hybrid_store = hybrid_store
        self.llm = llm_provider
        self.min_interactions_per_role = 10
        
    async def discover_roles_from_interactions(
        self,
        days_back: int = 30,
        max_roles: int = 5
    ) -> List[CommandRole]:
        """Discover roles by having LLM analyze interaction patterns."""
        logger.info(f"Starting LLM-based role discovery (last {days_back} days)")
        
        # Get recent interactions grouped by patterns
        interaction_groups = await self._group_similar_interactions(days_back)
        
        if not interaction_groups:
            logger.info("No interaction patterns found")
            return []
        
        # Have LLM analyze each group and create roles
        roles = []
        for group_name, interactions in interaction_groups.items():
            if len(interactions) < self.min_interactions_per_role:
                continue
                
            role = await self._create_role_from_interactions(
                group_name, 
                interactions
            )
            if role:
                roles.append(role)
                
            if len(roles) >= max_roles:
                break
        
        logger.info(f"Discovered {len(roles)} roles via LLM analysis")
        return roles
    
    async def _group_similar_interactions(
        self, 
        days_back: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group interactions by similarity for role analysis."""
        since = datetime.now() - timedelta(days=days_back)
        
        # Get all recent interactions by searching with a very low threshold
        # This is a workaround to get all interactions
        import numpy as np
        dummy_embedding = np.random.randn(768 if self.hybrid_store._llm_embeddings else 384)
        all_results = await self.hybrid_store.vector_store.search_similar(
            query_embedding=dummy_embedding,
            k=1000,
            threshold=-1.0
        )
        
        # Filter for user queries only
        results = [
            (r[0], r[1], r[2]) for r in all_results 
            if r[2].get('type') == 'user_query'
        ]
        
        # Group by action patterns and keywords
        groups = {}
        
        for _, _, metadata in results:
            if metadata.get('timestamp', '') < since.isoformat():
                continue
                
            # Create group key from actions and query type
            actions = metadata.get('actions', [])
            query = metadata.get('text', '').lower()
            
            # Simple grouping by first action and key terms
            if actions:
                key_action = actions[0]
                
                # Look for key terms
                if any(term in query for term in ['debug', 'fix', 'error', 'bug']):
                    group_key = f"{key_action}_debugging"
                elif any(term in query for term in ['create', 'add', 'implement', 'build']):
                    group_key = f"{key_action}_building"
                elif any(term in query for term in ['optimize', 'improve', 'speed', 'performance']):
                    group_key = f"{key_action}_optimizing"
                elif any(term in query for term in ['test', 'verify', 'check']):
                    group_key = f"{key_action}_testing"
                else:
                    group_key = f"{key_action}_general"
                
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(metadata)
        
        return groups
    
    async def _create_role_from_interactions(
        self,
        group_name: str,
        interactions: List[Dict[str, Any]]
    ) -> Optional[CommandRole]:
        """Have LLM create a role from interaction examples."""
        
        # Prepare interaction summaries for LLM
        interaction_summaries = []
        for i, interaction in enumerate(interactions[:20]):  # Limit to 20 examples
            summary = {
                'query': interaction.get('text', ''),
                'actions': interaction.get('actions', []),
                'success': interaction.get('success', False),
                'files': self._extract_file_patterns(interaction)
            }
            interaction_summaries.append(summary)
        
        # Create prompt for LLM following Claude best practices
        prompt = f"""Analyze these software development interactions and create a specialized role definition.

## Interaction Examples

{json.dumps(interaction_summaries, indent=2)}

## Task

Based on these interactions, create a role that would excel at handling similar tasks. The role should:

1. Have a clear, specific name (e.g., "API Endpoint Developer", "Python Debugger")
2. Include a detailed description of expertise
3. Define clear activation triggers (keywords, file patterns, action sequences)
4. Specify communication style (concise/detailed, proactive/reactive)
5. List preferred tools and decision rules

## Response Format

Respond with a JSON object:

{{
  "name": "Role Name",
  "description": "Detailed description of what this role specializes in",
  "triggers": {{
    "keywords": ["list", "of", "trigger", "words"],
    "file_patterns": ["*.py", "*.js"],
    "action_patterns": ["common->action->sequence"]
  }},
  "communication": {{
    "style": "concise|explanatory|interactive|autonomous",
    "verbosity": 0.5,
    "proactivity": 0.7
  }},
  "tools": {{
    "preferred": ["tool1", "tool2"],
    "weights": {{"tool1": 0.9, "tool2": 0.8}}
  }},
  "rules": [
    {{"condition": "when X happens", "action": "do Y"}},
    {{"condition": "if error found", "action": "analyze and fix"}}
  ],
  "confidence": 0.8
}}"""

        try:
            # Get LLM response
            response = await self.llm.generate(prompt)
            
            # Parse response
            if isinstance(response, str):
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    role_data = json.loads(json_match.group())
                else:
                    logger.error(f"No JSON found in LLM response: {response[:200]}")
                    return None
            else:
                role_data = response
            
            # Create CommandRole from LLM analysis
            return self._create_role_from_llm_data(role_data)
            
        except Exception as e:
            logger.error(f"Error creating role from LLM: {e}")
            return None
    
    def _create_role_from_llm_data(self, data: Dict[str, Any]) -> Optional[CommandRole]:
        """Convert LLM response to CommandRole object."""
        try:
            # Map communication style
            style_map = {
                'concise': CommunicationStyle.CONCISE,
                'explanatory': CommunicationStyle.EXPLANATORY,
                'interactive': CommunicationStyle.INTERACTIVE,
                'autonomous': CommunicationStyle.AUTONOMOUS
            }
            
            comm_data = data.get('communication', {})
            comm_style = style_map.get(
                comm_data.get('style', 'explanatory'),
                CommunicationStyle.EXPLANATORY
            )
            
            # Create triggers
            triggers = []
            trigger_data = data.get('triggers', {})
            
            # Keywords
            for keyword in trigger_data.get('keywords', []):
                triggers.append(ContextTrigger(
                    TriggerType.TASK_KEYWORD,
                    keyword,
                    0.8
                ))
            
            # File patterns
            for pattern in trigger_data.get('file_patterns', []):
                triggers.append(ContextTrigger(
                    TriggerType.FILE_PATTERN,
                    pattern.replace('*', '.*'),
                    0.7
                ))
            
            # Action patterns
            for pattern in trigger_data.get('action_patterns', []):
                triggers.append(ContextTrigger(
                    TriggerType.INTERACTION_PATTERN,
                    pattern,
                    0.9
                ))
            
            # Create decision rules
            rules = []
            for rule_data in data.get('rules', []):
                rules.append(DecisionRule(
                    condition=rule_data.get('condition', ''),
                    action=rule_data.get('action', ''),
                    priority=len(rules)
                ))
            
            # Create role
            role = CommandRole(
                name=data.get('name', f'Role_{datetime.now().timestamp()}'),
                description=data.get('description', 'Auto-discovered role'),
                context_triggers=triggers,
                communication_profile=CommunicationProfile(
                    style=comm_style,
                    verbosity=comm_data.get('verbosity', 0.5),
                    proactivity=comm_data.get('proactivity', 0.5)
                ),
                tool_preferences=data.get('tools', {}).get('weights', {}),
                decision_rules=rules,
                confidence_threshold=data.get('confidence', 0.7)
            )
            
            return role
            
        except Exception as e:
            logger.error(f"Error creating role from data: {e}")
            return None
    
    def _extract_file_patterns(self, interaction: Dict[str, Any]) -> List[str]:
        """Extract file patterns from interaction."""
        files = []
        
        # Look for file paths in various fields
        for field in ['target', 'file_path', 'files']:
            if field in interaction:
                value = interaction[field]
                if isinstance(value, str):
                    files.append(value)
                elif isinstance(value, list):
                    files.extend(value)
        
        # Extract patterns
        patterns = set()
        for file in files:
            if '.' in file:
                ext = file.split('.')[-1]
                patterns.add(f'*.{ext}')
        
        return list(patterns)