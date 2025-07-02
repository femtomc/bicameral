"""Enhanced role discovery using complete interaction model."""

import asyncio
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import re
import json

# Optional imports for advanced NLP
try:
    import numpy as np
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core.memory import Memory
from ..core.interaction_model import Interaction, Action, FeedbackType
from ..storage.hybrid_store import HybridStore
from ..utils.logging_config import get_logger
from .roles import (
    CommandRole, ContextTrigger, TriggerType,
    DecisionRule, CommunicationProfile, CommunicationStyle
)


logger = get_logger("interaction_role_discoverer")


class InteractionRoleDiscoverer:
    """Discovers behavioral roles from complete interaction patterns."""
    
    def __init__(self, memory: Memory, hybrid_store: HybridStore):
        self.memory = memory
        self.hybrid_store = hybrid_store
        self.min_cluster_size = 10  # Minimum interactions to form a role
        self.similarity_threshold = 0.65
        self.min_role_confidence = 0.7
        
    async def discover_roles_from_interactions(
        self, 
        days_back: int = 30,
        min_interactions: int = 50
    ) -> List[CommandRole]:
        """Discover roles from complete interaction history."""
        logger.info(f"Starting role discovery from interactions (last {days_back} days)")
        
        # Get recent complete interactions
        interactions = await self._get_recent_interactions(days_back)
        
        if len(interactions) < min_interactions:
            logger.info(f"Not enough interactions ({len(interactions)}) for role discovery")
            return []
        
        # Extract features from interactions
        feature_matrix, interaction_data = await self._extract_interaction_features(interactions)
        
        if feature_matrix is None:
            return []
        
        # Discover role clusters
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
                role = await self._extract_role_from_interactions(cluster_interactions)
                if role and role.confidence_threshold >= self.min_role_confidence:
                    roles.append(role)
        
        # Merge similar roles
        roles = self._merge_similar_roles(roles)
        
        logger.info(f"Discovered {len(roles)} behavioral roles")
        return roles
    
    async def _get_recent_interactions(self, days_back: int) -> List[Interaction]:
        """Get recent complete interactions from storage."""
        since = datetime.now() - timedelta(days=days_back)
        
        # Query hybrid store for recent interactions
        # This would ideally use a proper query method
        all_interactions = []
        
        # For now, use search to get recent interactions
        recent_results = await self.hybrid_store.search_similar_queries(
            "interaction",  # Generic search
            k=1000  # Get many results
        )
        
        for _, _, metadata in recent_results:
            if metadata.get('timestamp', '') >= since.isoformat():
                # Reconstruct interaction from metadata
                interaction = self._reconstruct_interaction(metadata)
                if interaction:
                    all_interactions.append(interaction)
        
        return all_interactions
    
    def _reconstruct_interaction(self, metadata: Dict[str, Any]) -> Optional[Interaction]:
        """Reconstruct an Interaction object from stored metadata."""
        try:
            interaction = Interaction(
                user_query=metadata.get('text', ''),
                query_context=metadata.get('context', {})
            )
            
            # Set properties
            interaction.interaction_id = metadata.get('id', '')
            interaction.timestamp = datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat()))
            interaction.ai_interpretation = metadata.get('interpretation', '')
            interaction.planned_actions = metadata.get('planned_actions', [])
            interaction.success = metadata.get('success', False)
            interaction.user_feedback = metadata.get('feedback', '')
            
            # Reconstruct actions
            for action_data in metadata.get('actions', []):
                action = Action(
                    action_type=action_data.get('type', ''),
                    target=action_data.get('target'),
                    details=action_data.get('details', {})
                )
                interaction.actions_taken.append(action)
            
            return interaction
        except Exception as e:
            logger.error(f"Error reconstructing interaction: {e}")
            return None
    
    async def _extract_interaction_features(
        self, 
        interactions: List[Interaction]
    ) -> Tuple[Optional[Any], List[Dict]]:
        """Extract rich features from complete interactions."""
        if not interactions:
            return None, []
        
        documents = []
        interaction_data = []
        
        for interaction in interactions:
            # Create comprehensive text representation
            text_parts = []
            
            # User query features
            query_words = self._extract_keywords(interaction.user_query)
            text_parts.extend([f"query:{word}" for word in query_words])
            
            # AI interpretation features
            if interaction.ai_interpretation:
                interp_words = self._extract_keywords(interaction.ai_interpretation)
                text_parts.extend([f"interpretation:{word}" for word in interp_words])
            
            # Action sequence features
            action_sequence = []
            for action in interaction.actions_taken:
                action_sequence.append(action.action_type)
                text_parts.append(f"action:{action.action_type}")
                
                # File type features
                if action.target and '.' in action.target:
                    ext = action.target.split('.')[-1]
                    text_parts.append(f"filetype:{ext}")
            
            # Action sequence pattern
            if len(action_sequence) >= 2:
                sequence_str = "->".join(action_sequence[:5])  # Limit length
                text_parts.append(f"sequence:{sequence_str}")
            
            # Success/feedback features
            text_parts.append(f"success:{interaction.success}")
            if interaction.feedback_type:
                text_parts.append(f"feedback:{interaction.feedback_type.value}")
            
            # Time features
            hour = interaction.timestamp.hour
            text_parts.append(f"hour:{hour}")
            text_parts.append(f"dayofweek:{interaction.timestamp.weekday()}")
            
            # Execution time features
            if interaction.execution_time:
                if interaction.execution_time < 1:
                    text_parts.append("speed:fast")
                elif interaction.execution_time < 5:
                    text_parts.append("speed:normal")
                else:
                    text_parts.append("speed:slow")
            
            document = " ".join(text_parts)
            documents.append(document)
            
            interaction_data.append({
                'interaction': interaction,
                'document': document,
                'query_keywords': query_words,
                'action_sequence': action_sequence,
                'success_rate': 1.0 if interaction.success else 0.0
            })
        
        # Use advanced features if available
        if SKLEARN_AVAILABLE:
            try:
                # TF-IDF with bi-grams
                vectorizer = TfidfVectorizer(
                    max_features=200,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.8
                )
                feature_matrix = vectorizer.fit_transform(documents)
                return feature_matrix.toarray(), interaction_data
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                return documents, interaction_data
        else:
            return documents, interaction_data
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Simple keyword extraction
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter and return
        keywords = []
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)
        
        return keywords[:10]  # Limit to top 10
    
    def _cluster_interactions(self, feature_matrix: Any) -> List[int]:
        """Cluster interactions using appropriate algorithm."""
        if SKLEARN_AVAILABLE and isinstance(feature_matrix, np.ndarray):
            # Try DBSCAN first for density-based clustering
            try:
                clustering = DBSCAN(
                    eps=0.35,
                    min_samples=self.min_cluster_size,
                    metric='cosine'
                )
                clusters = clustering.fit_predict(feature_matrix)
                
                # If too few clusters, try KMeans
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                if n_clusters < 3:
                    # Estimate optimal number of clusters
                    n_samples = feature_matrix.shape[0]
                    optimal_k = min(max(3, n_samples // 50), 10)
                    
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                    clusters = kmeans.fit_predict(feature_matrix)
                
                return clusters
            except Exception as e:
                logger.error(f"Clustering error: {e}")
                return self._simple_clustering(feature_matrix)
        else:
            return self._simple_clustering(feature_matrix)
    
    def _simple_clustering(self, documents: List[str]) -> List[int]:
        """Enhanced simple clustering for when sklearn is not available."""
        clusters = [-1] * len(documents)
        cluster_id = 0
        
        # Group by interaction patterns
        pattern_groups = defaultdict(list)
        
        for i, doc in enumerate(documents):
            # Extract key patterns
            patterns = []
            
            # Action sequences
            seq_match = re.search(r'sequence:([^\s]+)', doc)
            if seq_match:
                patterns.append(('seq', seq_match.group(1)))
            
            # Query keywords
            query_keywords = re.findall(r'query:(\w+)', doc)
            if query_keywords:
                patterns.append(('query', '-'.join(sorted(query_keywords[:3]))))
            
            # Success pattern
            success_match = re.search(r'success:(\w+)', doc)
            if success_match:
                patterns.append(('success', success_match.group(1)))
            
            # Create pattern key
            pattern_key = '|'.join([f"{k}:{v}" for k, v in patterns])
            if pattern_key:
                pattern_groups[pattern_key].append(i)
        
        # Assign clusters to groups
        for pattern_key, indices in pattern_groups.items():
            if len(indices) >= self.min_cluster_size:
                for idx in indices:
                    clusters[idx] = cluster_id
                cluster_id += 1
        
        return clusters
    
    async def _extract_role_from_interactions(
        self, 
        cluster_data: List[Dict[str, Any]]
    ) -> Optional[CommandRole]:
        """Extract a role from a cluster of similar interactions."""
        if not cluster_data:
            return None
        
        # Analyze cluster characteristics
        interactions = [d['interaction'] for d in cluster_data]
        
        # Analyze queries
        query_patterns = self._analyze_query_patterns(interactions)
        
        # Analyze action sequences
        action_patterns = self._analyze_action_patterns(interactions)
        
        # Analyze success patterns
        success_metrics = self._analyze_success_patterns(interactions)
        
        # Analyze timing patterns
        timing_patterns = self._analyze_timing_patterns(interactions)
        
        # Generate role characteristics
        role_name = self._generate_role_name(query_patterns, action_patterns)
        role_description = self._generate_role_description(
            query_patterns, action_patterns, success_metrics
        )
        
        # Create context triggers
        triggers = self._create_context_triggers(
            query_patterns, action_patterns, timing_patterns
        )
        
        if not triggers:
            return None
        
        # Infer tool preferences
        tool_preferences = self._infer_tool_preferences(action_patterns)
        
        # Generate decision rules
        decision_rules = self._generate_decision_rules(
            query_patterns, action_patterns, success_metrics
        )
        
        # Determine communication style
        comm_profile = self._determine_communication_profile(interactions)
        
        # Calculate role confidence
        confidence = self._calculate_role_confidence(
            len(interactions), success_metrics['success_rate']
        )
        
        return CommandRole(
            name=role_name,
            description=role_description,
            context_triggers=triggers,
            confidence_threshold=confidence,
            tool_preferences=tool_preferences,
            decision_rules=decision_rules,
            communication_profile=comm_profile,
            success_rate=success_metrics['success_rate'],
            usage_count=0,
            successful_patterns=action_patterns['common_sequences'][:3]
        )
    
    def _analyze_query_patterns(self, interactions: List[Interaction]) -> Dict[str, Any]:
        """Analyze patterns in user queries."""
        all_keywords = []
        query_types = Counter()
        
        for interaction in interactions:
            keywords = self._extract_keywords(interaction.user_query)
            all_keywords.extend(keywords)
            
            # Classify query type
            query_lower = interaction.user_query.lower()
            if any(word in query_lower for word in ['fix', 'debug', 'error', 'bug']):
                query_types['debugging'] += 1
            elif any(word in query_lower for word in ['create', 'implement', 'add', 'build']):
                query_types['creation'] += 1
            elif any(word in query_lower for word in ['refactor', 'improve', 'optimize']):
                query_types['refactoring'] += 1
            elif any(word in query_lower for word in ['test', 'verify', 'check']):
                query_types['testing'] += 1
            elif any(word in query_lower for word in ['understand', 'explain', 'what', 'how']):
                query_types['exploration'] += 1
            else:
                query_types['general'] += 1
        
        # Get most common keywords
        keyword_counter = Counter(all_keywords)
        common_keywords = [kw for kw, _ in keyword_counter.most_common(10)]
        
        return {
            'common_keywords': common_keywords,
            'query_types': dict(query_types),
            'dominant_type': query_types.most_common(1)[0][0] if query_types else 'general'
        }
    
    def _analyze_action_patterns(self, interactions: List[Interaction]) -> Dict[str, Any]:
        """Analyze patterns in action sequences."""
        action_sequences = []
        action_counter = Counter()
        file_patterns = Counter()
        
        for interaction in interactions:
            sequence = [a.action_type for a in interaction.actions_taken]
            if sequence:
                action_sequences.append(sequence)
                
                for action in interaction.actions_taken:
                    action_counter[action.action_type] += 1
                    
                    if action.target and '.' in action.target:
                        ext = action.target.split('.')[-1]
                        file_patterns[ext] += 1
        
        # Find common subsequences
        common_sequences = self._find_common_subsequences(action_sequences)
        
        return {
            'common_actions': [a for a, _ in action_counter.most_common(5)],
            'common_sequences': common_sequences,
            'file_types': [f for f, _ in file_patterns.most_common(3)],
            'avg_sequence_length': sum(len(s) for s in action_sequences) / len(action_sequences) if action_sequences else 0
        }
    
    def _find_common_subsequences(self, sequences: List[List[str]]) -> List[str]:
        """Find common subsequences in action patterns."""
        if not sequences:
            return []
        
        # Find 2-grams and 3-grams
        ngram_counter = Counter()
        
        for sequence in sequences:
            # 2-grams
            for i in range(len(sequence) - 1):
                ngram = "->".join(sequence[i:i+2])
                ngram_counter[ngram] += 1
            
            # 3-grams
            for i in range(len(sequence) - 2):
                ngram = "->".join(sequence[i:i+3])
                ngram_counter[ngram] += 1
        
        # Return most common patterns that appear in at least 30% of sequences
        min_count = len(sequences) * 0.3
        common_patterns = [
            pattern for pattern, count in ngram_counter.most_common(5)
            if count >= min_count
        ]
        
        return common_patterns
    
    def _analyze_success_patterns(self, interactions: List[Interaction]) -> Dict[str, Any]:
        """Analyze success and feedback patterns."""
        success_count = sum(1 for i in interactions if i.success)
        total_count = len(interactions)
        
        feedback_types = Counter()
        for interaction in interactions:
            if interaction.feedback_type:
                feedback_types[interaction.feedback_type.value] += 1
        
        # Analyze what makes interactions successful
        successful_actions = Counter()
        failed_actions = Counter()
        
        for interaction in interactions:
            action_types = [a.action_type for a in interaction.actions_taken]
            if interaction.success:
                successful_actions.update(action_types)
            else:
                failed_actions.update(action_types)
        
        return {
            'success_rate': success_count / total_count if total_count > 0 else 0,
            'feedback_distribution': dict(feedback_types),
            'successful_actions': [a for a, _ in successful_actions.most_common(3)],
            'failed_actions': [a for a, _ in failed_actions.most_common(3)]
        }
    
    def _analyze_timing_patterns(self, interactions: List[Interaction]) -> Dict[str, Any]:
        """Analyze timing patterns."""
        hour_distribution = Counter()
        day_distribution = Counter()
        execution_times = []
        
        for interaction in interactions:
            hour_distribution[interaction.timestamp.hour] += 1
            day_distribution[interaction.timestamp.weekday()] += 1
            
            if interaction.execution_time:
                execution_times.append(interaction.execution_time)
        
        return {
            'peak_hours': [h for h, _ in hour_distribution.most_common(3)],
            'peak_days': [d for d, _ in day_distribution.most_common(2)],
            'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0
        }
    
    def _generate_role_name(
        self, 
        query_patterns: Dict[str, Any],
        action_patterns: Dict[str, Any]
    ) -> str:
        """Generate a descriptive role name."""
        dominant_query_type = query_patterns['dominant_type']
        common_actions = action_patterns['common_actions']
        
        # Role name templates based on patterns
        if dominant_query_type == 'debugging':
            return f"Debug Specialist ({common_actions[0] if common_actions else 'General'})"
        elif dominant_query_type == 'creation':
            return f"Feature Builder ({common_actions[0] if common_actions else 'General'})"
        elif dominant_query_type == 'refactoring':
            return "Code Improver"
        elif dominant_query_type == 'testing':
            return "Test Engineer"
        elif dominant_query_type == 'exploration':
            return "Code Explorer"
        else:
            # Generate based on common actions
            if common_actions:
                return f"{common_actions[0].title()} Specialist"
            else:
                return "General Developer"
    
    def _generate_role_description(
        self,
        query_patterns: Dict[str, Any],
        action_patterns: Dict[str, Any],
        success_metrics: Dict[str, Any]
    ) -> str:
        """Generate a role description."""
        query_keywords = ", ".join(query_patterns['common_keywords'][:5])
        success_rate = success_metrics['success_rate']
        avg_actions = action_patterns['avg_sequence_length']
        
        return (
            f"Specializes in handling queries about {query_keywords}. "
            f"Typically performs {avg_actions:.1f} actions per task with "
            f"{success_rate:.0%} success rate."
        )
    
    def _create_context_triggers(
        self,
        query_patterns: Dict[str, Any],
        action_patterns: Dict[str, Any],
        timing_patterns: Dict[str, Any]
    ) -> List[ContextTrigger]:
        """Create context triggers for the role."""
        triggers = []
        
        # Query keyword triggers
        if query_patterns['common_keywords']:
            triggers.append(ContextTrigger(
                trigger_type=TriggerType.TASK_KEYWORD,
                pattern=",".join(query_patterns['common_keywords'][:7]),
                weight=0.8
            ))
        
        # Action sequence triggers
        if action_patterns['common_sequences']:
            for sequence in action_patterns['common_sequences'][:2]:
                triggers.append(ContextTrigger(
                    trigger_type=TriggerType.INTERACTION_PATTERN,
                    pattern=sequence,
                    weight=0.7
                ))
        
        # File pattern triggers
        if action_patterns['file_types']:
            for file_type in action_patterns['file_types'][:2]:
                triggers.append(ContextTrigger(
                    trigger_type=TriggerType.FILE_PATTERN,
                    pattern=f".*\\.{file_type}$",
                    weight=0.6
                ))
        
        # Time-based triggers (if strong pattern)
        peak_hours = timing_patterns['peak_hours']
        if peak_hours and len(set(peak_hours)) <= 2:  # Concentrated in few hours
            triggers.append(ContextTrigger(
                trigger_type=TriggerType.TIME_BASED,
                pattern=f"hour:{peak_hours[0]}",
                weight=0.4
            ))
        
        return triggers
    
    def _infer_tool_preferences(self, action_patterns: Dict[str, Any]) -> Dict[str, float]:
        """Infer tool preferences from action patterns."""
        # Map actions to tools
        action_tool_map = {
            'search': ['search_memory', 'get_relevant_context'],
            'read': ['get_memory_insights', 'get_relevant_context'],
            'edit': ['log_action', 'complete_interaction'],
            'test': ['log_action', 'record_feedback'],
            'analyze': ['detect_pattern', 'get_memory_stats'],
            'implement': ['start_interaction', 'log_action', 'complete_interaction'],
            'debug': ['search_memory', 'log_action', 'record_feedback'],
            'refactor': ['get_relevant_context', 'detect_pattern', 'log_action']
        }
        
        tool_weights = defaultdict(float)
        
        # Weight tools based on common actions
        for i, action in enumerate(action_patterns['common_actions']):
            weight = 1.0 - (i * 0.15)  # Decreasing weight
            
            # Find matching tools
            for action_key, tools in action_tool_map.items():
                if action_key in action.lower():
                    for tool in tools:
                        tool_weights[tool] += weight
        
        # Normalize weights
        if tool_weights:
            max_weight = max(tool_weights.values())
            return {tool: weight / max_weight for tool, weight in tool_weights.items()}
        
        return {}
    
    def _generate_decision_rules(
        self,
        query_patterns: Dict[str, Any],
        action_patterns: Dict[str, Any],
        success_metrics: Dict[str, Any]
    ) -> List[DecisionRule]:
        """Generate decision rules based on patterns."""
        rules = []
        
        # Rules based on query type
        dominant_type = query_patterns['dominant_type']
        
        if dominant_type == 'debugging':
            rules.append(DecisionRule(
                condition="encountering an error",
                action="analyze the error context before attempting fixes",
                priority=10
            ))
            rules.append(DecisionRule(
                condition="debugging issues",
                action="search for similar past errors first",
                priority=9
            ))
        elif dominant_type == 'creation':
            rules.append(DecisionRule(
                condition="implementing new features",
                action="check for existing patterns and examples",
                priority=9
            ))
            rules.append(DecisionRule(
                condition="creating new code",
                action="follow established project conventions",
                priority=8
            ))
        elif dominant_type == 'refactoring':
            rules.append(DecisionRule(
                condition="refactoring code",
                action="ensure tests pass before and after changes",
                priority=10
            ))
        
        # Rules based on success patterns
        if success_metrics['successful_actions']:
            action = success_metrics['successful_actions'][0]
            rules.append(DecisionRule(
                condition="working on similar tasks",
                action=f"prioritize {action} actions based on past success",
                priority=7
            ))
        
        # Rules based on failure patterns
        if success_metrics['failed_actions']:
            action = success_metrics['failed_actions'][0]
            rules.append(DecisionRule(
                condition=f"considering {action} actions",
                action="be extra careful as these have failed before",
                priority=6
            ))
        
        # Add general rule
        if not rules:
            rules.append(DecisionRule(
                condition="handling this type of task",
                action="follow patterns from similar successful interactions",
                priority=5
            ))
        
        return rules
    
    def _determine_communication_profile(
        self, 
        interactions: List[Interaction]
    ) -> CommunicationProfile:
        """Determine communication style from interactions."""
        # Analyze interaction characteristics
        total = len(interactions)
        if total == 0:
            return CommunicationProfile(style=CommunicationStyle.CONCISE)
        
        # Count characteristics
        long_queries = sum(1 for i in interactions if len(i.user_query.split()) > 20)
        detailed_feedback = sum(1 for i in interactions if i.user_feedback and len(i.user_feedback) > 50)
        multi_action = sum(1 for i in interactions if len(i.actions_taken) > 3)
        questions = sum(1 for i in interactions if '?' in i.user_query)
        
        # Determine style
        if questions / total > 0.4:
            style = CommunicationStyle.INTERACTIVE
        elif long_queries / total > 0.3 or detailed_feedback / total > 0.3:
            style = CommunicationStyle.EXPLANATORY
        elif multi_action / total > 0.5:
            style = CommunicationStyle.AUTONOMOUS
        else:
            style = CommunicationStyle.CONCISE
        
        # Calculate other metrics
        verbosity = min(1.0, (long_queries + detailed_feedback) / (total * 2))
        proactivity = min(1.0, multi_action / total)
        question_frequency = questions / total
        
        return CommunicationProfile(
            style=style,
            verbosity=verbosity,
            proactivity=proactivity,
            question_frequency=question_frequency
        )
    
    def _calculate_role_confidence(self, interaction_count: int, success_rate: float) -> float:
        """Calculate confidence threshold for the role."""
        # Base confidence on data volume and success
        base_confidence = 0.5
        
        # Adjust based on interaction count
        if interaction_count > 100:
            base_confidence += 0.2
        elif interaction_count > 50:
            base_confidence += 0.1
        
        # Adjust based on success rate
        if success_rate > 0.8:
            base_confidence += 0.1
        elif success_rate < 0.5:
            base_confidence -= 0.1
        
        return max(0.3, min(0.9, base_confidence))
    
    def _merge_similar_roles(self, roles: List[CommandRole]) -> List[CommandRole]:
        """Merge roles that are too similar."""
        if len(roles) <= 1:
            return roles
        
        merged_roles = []
        used_indices = set()
        
        for i, role1 in enumerate(roles):
            if i in used_indices:
                continue
                
            # Check for similar roles
            similar_roles = [role1]
            for j, role2 in enumerate(roles[i+1:], start=i+1):
                if j in used_indices:
                    continue
                    
                if self._are_roles_similar(role1, role2):
                    similar_roles.append(role2)
                    used_indices.add(j)
            
            # Merge if multiple similar roles
            if len(similar_roles) > 1:
                merged_role = self._merge_role_group(similar_roles)
                merged_roles.append(merged_role)
            else:
                merged_roles.append(role1)
        
        return merged_roles
    
    def _are_roles_similar(self, role1: CommandRole, role2: CommandRole) -> bool:
        """Check if two roles are similar enough to merge."""
        # Compare triggers
        triggers1 = {(t.trigger_type, t.pattern) for t in role1.context_triggers}
        triggers2 = {(t.trigger_type, t.pattern) for t in role2.context_triggers}
        
        if not triggers1 or not triggers2:
            return False
        
        overlap = len(triggers1 & triggers2)
        union = len(triggers1 | triggers2)
        
        trigger_similarity = overlap / union if union > 0 else 0
        
        # Compare tool preferences
        tools1 = set(role1.tool_preferences.keys())
        tools2 = set(role2.tool_preferences.keys())
        
        if tools1 and tools2:
            tool_overlap = len(tools1 & tools2)
            tool_union = len(tools1 | tools2)
            tool_similarity = tool_overlap / tool_union if tool_union > 0 else 0
        else:
            tool_similarity = 0
        
        # Similar if high overlap in triggers or tools
        return trigger_similarity > 0.6 or tool_similarity > 0.7
    
    def _merge_role_group(self, roles: List[CommandRole]) -> CommandRole:
        """Merge a group of similar roles into one."""
        # Use the most successful role as base
        base_role = max(roles, key=lambda r: r.success_rate)
        
        # Merge triggers (union)
        all_triggers = []
        trigger_set = set()
        for role in roles:
            for trigger in role.context_triggers:
                key = (trigger.trigger_type, trigger.pattern)
                if key not in trigger_set:
                    trigger_set.add(key)
                    all_triggers.append(trigger)
        
        # Merge tool preferences (weighted average)
        merged_tools = defaultdict(float)
        total_weight = sum(r.usage_count + 1 for r in roles)
        
        for role in roles:
            weight = (role.usage_count + 1) / total_weight
            for tool, pref in role.tool_preferences.items():
                merged_tools[tool] += pref * weight
        
        # Merge decision rules (union with priority adjustment)
        all_rules = []
        for role in roles:
            all_rules.extend(role.decision_rules)
        
        # Sort by priority and take top rules
        all_rules.sort(key=lambda r: r.priority, reverse=True)
        merged_rules = all_rules[:10]  # Limit to 10 rules
        
        # Create merged role
        merged_role = CommandRole(
            name=f"{base_role.name} (Merged)",
            description=f"{base_role.description} [Merged from {len(roles)} similar roles]",
            context_triggers=all_triggers,
            confidence_threshold=sum(r.confidence_threshold for r in roles) / len(roles),
            tool_preferences=dict(merged_tools),
            decision_rules=merged_rules,
            communication_profile=base_role.communication_profile,
            success_rate=sum(r.success_rate for r in roles) / len(roles),
            usage_count=sum(r.usage_count for r in roles)
        )
        
        return merged_role