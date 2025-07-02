"""Role discovery and mining from interaction patterns."""

import asyncio
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import re

# Optional imports for advanced clustering
try:
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core.memory import Memory
from ..utils.logging_config import get_logger
from .roles import (
    CommandRole, ContextTrigger, TriggerType, 
    DecisionRule, CommunicationProfile, CommunicationStyle
)


logger = get_logger("role_discoverer")


class RoleDiscoverer:
    """Discovers and extracts behavioral roles from interaction patterns."""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.min_cluster_size = 5  # Minimum interactions to form a role
        self.similarity_threshold = 0.7
        
    async def discover_roles(self, days_back: int = 30) -> List[CommandRole]:
        """Discover new roles from recent interactions."""
        logger.info(f"Starting role discovery for last {days_back} days")
        
        # Get recent interactions
        since = datetime.now() - timedelta(days=days_back)
        interactions = await self._get_interactions_since(since)
        
        if len(interactions) < self.min_cluster_size:
            logger.info(f"Not enough interactions ({len(interactions)}) for role discovery")
            return []
        
        # Extract features from interactions
        feature_matrix, interaction_data = self._extract_features(interactions)
        
        if feature_matrix is None:
            return []
        
        # Cluster similar interactions
        clusters = self._cluster_interactions(feature_matrix)
        
        # Extract roles from clusters
        roles = []
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Noise points
                continue
                
            cluster_interactions = [
                interaction_data[i] 
                for i, c in enumerate(clusters) 
                if c == cluster_id
            ]
            
            if len(cluster_interactions) >= self.min_cluster_size:
                role = self._extract_role_from_cluster(cluster_interactions)
                if role:
                    roles.append(role)
        
        logger.info(f"Discovered {len(roles)} potential roles")
        return roles
    
    async def _get_interactions_since(self, since: datetime) -> List[Dict[str, Any]]:
        """Get all interactions since a given date."""
        # This would query the database for interactions
        # For now, using the memory manager's recent interactions
        all_interactions = []
        
        # Get recent interactions (last 1000)
        recent = await self.memory.get_recent_interactions(limit=1000)
        
        for interaction in recent:
            if interaction['timestamp'] >= since.isoformat():
                all_interactions.append(interaction)
        
        return all_interactions
    
    def _extract_features(self, interactions: List[Dict[str, Any]]) -> Tuple[Optional[Any], List[Dict]]:
        """Extract features from interactions for clustering."""
        if not interactions:
            return None, []
        
        # Prepare text data for vectorization
        documents = []
        interaction_data = []
        
        for interaction in interactions:
            # Create a text representation of the interaction
            text_parts = []
            
            # Add action
            action = interaction.get('action', '')
            text_parts.append(f"action:{action}")
            
            # Add file type if present
            file_path = interaction.get('file_path', '')
            if file_path:
                ext = file_path.split('.')[-1] if '.' in file_path else 'unknown'
                text_parts.append(f"filetype:{ext}")
                
                # Add directory context
                if '/' in file_path:
                    dir_name = file_path.split('/')[-2] if len(file_path.split('/')) > 1 else 'root'
                    text_parts.append(f"directory:{dir_name}")
            
            # Add details keywords
            details = interaction.get('details', {})
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, str):
                        text_parts.append(f"{key}:{value[:50]}")  # Limit length
            
            # Add session context
            session_id = interaction.get('session_id', 'unknown')
            text_parts.append(f"session:{session_id}")
            
            document = " ".join(text_parts)
            documents.append(document)
            
            interaction_data.append({
                'action': action,
                'file_path': file_path,
                'details': details,
                'timestamp': interaction.get('timestamp'),
                'session_id': session_id,
                'document': document
            })
        
        # If sklearn is available, use TF-IDF vectorization
        if SKLEARN_AVAILABLE:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    ngram_range=(1, 2),
                    stop_words='english'
                )
                feature_matrix = vectorizer.fit_transform(documents)
                return feature_matrix.toarray(), interaction_data
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                return None, []
        else:
            # Simple feature extraction without sklearn
            # Return documents and interaction data for simple clustering
            return documents, interaction_data
    
    def _cluster_interactions(self, feature_matrix: Any) -> List[int]:
        """Cluster interactions using DBSCAN or simple clustering."""
        if SKLEARN_AVAILABLE and isinstance(feature_matrix, np.ndarray):
            # Use DBSCAN for clustering
            clustering = DBSCAN(
                eps=0.3,  # Maximum distance between samples
                min_samples=self.min_cluster_size,
                metric='cosine'
            )
            
            clusters = clustering.fit_predict(feature_matrix)
            
            # Log cluster statistics
            unique_clusters = set(clusters)
            logger.info(f"Found {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} clusters")
            
            return clusters
        else:
            # Simple clustering based on document similarity
            documents = feature_matrix  # In this case, it's the list of documents
            clusters = self._simple_clustering(documents)
            return clusters
    
    def _simple_clustering(self, documents: List[str]) -> List[int]:
        """Simple clustering based on document similarity without sklearn."""
        clusters = [-1] * len(documents)  # Initialize all as noise
        cluster_id = 0
        
        # Group by similar documents
        for i, doc1 in enumerate(documents):
            if clusters[i] != -1:  # Already assigned
                continue
                
            # Find similar documents
            similar_indices = [i]
            doc1_words = set(doc1.lower().split())
            
            for j, doc2 in enumerate(documents[i+1:], start=i+1):
                if clusters[j] != -1:  # Already assigned
                    continue
                    
                doc2_words = set(doc2.lower().split())
                # Simple Jaccard similarity
                intersection = len(doc1_words & doc2_words)
                union = len(doc1_words | doc2_words)
                
                if union > 0 and intersection / union > self.similarity_threshold:
                    similar_indices.append(j)
            
            # Assign cluster if we have enough similar documents
            if len(similar_indices) >= self.min_cluster_size:
                for idx in similar_indices:
                    clusters[idx] = cluster_id
                cluster_id += 1
        
        return clusters
    
    def _extract_role_from_cluster(self, cluster_interactions: List[Dict[str, Any]]) -> Optional[CommandRole]:
        """Extract a role definition from a cluster of similar interactions."""
        if not cluster_interactions:
            return None
        
        # Analyze common patterns in the cluster
        action_counter = Counter()
        file_patterns = []
        keywords = set()
        success_count = 0
        total_count = len(cluster_interactions)
        
        for interaction in cluster_interactions:
            # Count actions
            action = interaction.get('action', '')
            action_counter[action] += 1
            
            # Collect file patterns
            file_path = interaction.get('file_path', '')
            if file_path:
                file_patterns.append(file_path)
            
            # Extract keywords from details
            details = interaction.get('details', {})
            if isinstance(details, dict):
                for key, value in details.items():
                    keywords.add(key)
                    if isinstance(value, str):
                        # Extract words from value
                        words = re.findall(r'\w+', value.lower())
                        keywords.update(words[:5])  # Limit to avoid noise
            
            # Track success (simplified - in real implementation would check outcomes)
            success_count += 1  # Assume success for now
        
        # Determine role characteristics
        common_actions = [action for action, count in action_counter.most_common(3)]
        
        # Infer role name and description
        role_name = self._generate_role_name(common_actions, file_patterns, keywords)
        role_description = self._generate_role_description(common_actions, keywords)
        
        # Create context triggers
        triggers = []
        
        # Add keyword triggers if we have consistent keywords
        if keywords:
            common_keywords = list(keywords)[:10]  # Top keywords
            triggers.append(ContextTrigger(
                trigger_type=TriggerType.TASK_KEYWORD,
                pattern=",".join(common_keywords),
                weight=0.8
            ))
        
        # Add file pattern trigger if consistent
        if file_patterns:
            # Find common file extension
            extensions = [fp.split('.')[-1] for fp in file_patterns if '.' in fp]
            if extensions:
                ext_counter = Counter(extensions)
                common_ext = ext_counter.most_common(1)[0][0]
                if ext_counter[common_ext] > len(file_patterns) * 0.5:  # >50% same extension
                    triggers.append(ContextTrigger(
                        trigger_type=TriggerType.FILE_PATTERN,
                        pattern=f".*\\.{common_ext}$",
                        weight=0.6
                    ))
        
        # Add interaction pattern trigger
        if len(common_actions) >= 2:
            pattern = "->".join(common_actions[:3])
            triggers.append(ContextTrigger(
                trigger_type=TriggerType.INTERACTION_PATTERN,
                pattern=pattern,
                weight=0.7
            ))
        
        if not triggers:
            return None  # Can't create role without triggers
        
        # Determine communication style based on interaction patterns
        comm_style = self._infer_communication_style(cluster_interactions)
        
        # Create the role
        role = CommandRole(
            name=role_name,
            description=role_description,
            context_triggers=triggers,
            confidence_threshold=0.6,  # Start with moderate threshold
            tool_preferences=self._infer_tool_preferences(common_actions),
            decision_rules=self._generate_decision_rules(common_actions, keywords),
            communication_profile=comm_style,
            success_rate=success_count / total_count,
            usage_count=0  # Will be updated as role is used
        )
        
        return role
    
    def _generate_role_name(self, actions: List[str], file_patterns: List[str], keywords: Set[str]) -> str:
        """Generate a descriptive role name."""
        # Simple heuristic-based naming
        if 'test' in keywords or 'test' in str(actions):
            return "Test Specialist"
        elif 'debug' in keywords or 'fix' in keywords:
            return "Debug Specialist"
        elif 'refactor' in keywords or 'improve' in keywords:
            return "Code Improver"
        elif 'create' in keywords or 'implement' in keywords:
            return "Feature Builder"
        elif any('config' in fp or 'settings' in fp for fp in file_patterns):
            return "Configuration Manager"
        elif 'document' in keywords or 'docs' in str(actions):
            return "Documentation Expert"
        else:
            # Generic name based on most common action
            action = actions[0] if actions else "General"
            return f"{action.title()} Specialist"
    
    def _generate_role_description(self, actions: List[str], keywords: Set[str]) -> str:
        """Generate a role description."""
        action_str = ", ".join(actions[:3]) if actions else "various tasks"
        keyword_str = ", ".join(list(keywords)[:5]) if keywords else "general operations"
        
        return f"Specializes in {action_str} with focus on {keyword_str}"
    
    def _infer_communication_style(self, interactions: List[Dict[str, Any]]) -> CommunicationProfile:
        """Infer communication style from interaction patterns."""
        # Analyze interaction patterns to determine style
        # This is simplified - real implementation would analyze actual communication
        
        total = len(interactions)
        if total == 0:
            return CommunicationProfile(style=CommunicationStyle.EXPLANATORY)
        
        # Count different types of actions
        question_count = sum(1 for i in interactions if 'question' in str(i.get('details', '')).lower())
        explanation_count = sum(1 for i in interactions if 'explain' in str(i.get('details', '')).lower())
        
        # Determine style based on patterns
        if question_count / total > 0.3:
            style = CommunicationStyle.INTERACTIVE
        elif explanation_count / total > 0.3:
            style = CommunicationStyle.EXPLANATORY
        elif total > 20:  # Many interactions suggest autonomous work
            style = CommunicationStyle.AUTONOMOUS
        else:
            style = CommunicationStyle.CONCISE
        
        return CommunicationProfile(
            style=style,
            verbosity=0.5,
            proactivity=0.6,
            question_frequency=question_count / total
        )
    
    def _infer_tool_preferences(self, actions: List[str]) -> Dict[str, float]:
        """Infer tool preferences from common actions."""
        # Map actions to tools (simplified)
        action_to_tools = {
            'search': ['search_memory', 'get_relevant_context'],
            'edit': ['log_interaction', 'detect_pattern'],
            'read': ['get_memory_insights', 'search_memory'],
            'test': ['log_interaction', 'record_feedback'],
            'analyze': ['get_memory_stats', 'detect_pattern'],
            'create': ['log_interaction', 'get_relevant_context']
        }
        
        tool_weights = defaultdict(float)
        
        for i, action in enumerate(actions):
            weight = 1.0 - (i * 0.2)  # Decrease weight for less common actions
            for tool in action_to_tools.get(action.lower(), []):
                tool_weights[tool] += weight
        
        # Normalize weights
        if tool_weights:
            max_weight = max(tool_weights.values())
            return {tool: weight / max_weight for tool, weight in tool_weights.items()}
        
        return {}
    
    def _generate_decision_rules(self, actions: List[str], keywords: Set[str]) -> List[DecisionRule]:
        """Generate decision rules based on patterns."""
        rules = []
        
        # Generate rules based on common patterns
        if 'test' in actions or 'test' in keywords:
            rules.append(DecisionRule(
                condition="making changes to code",
                action="run tests to verify behavior",
                priority=9
            ))
        
        if 'debug' in actions or 'error' in keywords:
            rules.append(DecisionRule(
                condition="encountering an error",
                action="gather context before attempting fixes",
                priority=10
            ))
        
        if 'refactor' in actions:
            rules.append(DecisionRule(
                condition="refactoring code",
                action="preserve existing behavior while improving structure",
                priority=8
            ))
        
        # Add a general rule
        if not rules:
            rules.append(DecisionRule(
                condition="working on this type of task",
                action="follow established patterns from similar past work",
                priority=5
            ))
        
        return rules
    
    async def evaluate_role_effectiveness(self, role: CommandRole, test_interactions: List[Dict[str, Any]]) -> float:
        """Evaluate how well a role would perform on test interactions."""
        if not test_interactions:
            return 0.0
        
        matches = 0
        for interaction in test_interactions:
            context = {
                'task_description': interaction.get('action', ''),
                'files': [interaction.get('file_path', '')],
                'recent_actions': [interaction.get('action', '')]
            }
            
            if role.should_activate(context):
                # In real implementation, would check if role's behavior matches interaction
                matches += 1
        
        return matches / len(test_interactions)
    
    async def refine_role(self, role: CommandRole, feedback: List[Dict[str, Any]]) -> CommandRole:
        """Refine a role based on feedback and performance data."""
        # Update success metrics
        for fb in feedback:
            success = fb.get('success', True)
            role.update_metrics(success)
        
        # Adjust confidence threshold based on performance
        if role.success_rate < 0.5 and role.usage_count > 10:
            # Increase threshold if role is not performing well
            role.confidence_threshold = min(0.9, role.confidence_threshold + 0.1)
        elif role.success_rate > 0.8 and role.usage_count > 20:
            # Decrease threshold if role is performing very well
            role.confidence_threshold = max(0.5, role.confidence_threshold - 0.05)
        
        # Could also adjust triggers, rules, etc. based on patterns in feedback
        
        return role