"""Pattern detection for complete interactions including natural language patterns."""

from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
import re
from datetime import datetime, timedelta

from .interaction_model import Interaction, InteractionPattern, FeedbackType
from ..utils.logging_config import get_logger

logger = get_logger("interaction_pattern_detector")


class InteractionPatternDetector:
    """Detects patterns in complete user interactions."""
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.min_frequency = 2  # Lower threshold for interaction patterns
        self.similarity_threshold = 0.7
        
        # Simple NLP components (no heavy dependencies)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'can',
            'could', 'please', 'would', 'should', 'will', 'this', 'that', 'these'
        }
        
    async def detect_patterns(self, interactions: List[Interaction]) -> List[InteractionPattern]:
        """Detect various patterns from interactions."""
        patterns = []
        
        # Different pattern detection strategies
        patterns.extend(await self._detect_intent_patterns(interactions))
        patterns.extend(await self._detect_success_patterns(interactions))
        patterns.extend(await self._detect_correction_patterns(interactions))
        patterns.extend(await self._detect_workflow_patterns(interactions))
        
        return patterns
    
    async def _detect_intent_patterns(self, interactions: List[Interaction]) -> List[InteractionPattern]:
        """Detect patterns in how users express similar intents."""
        patterns = []
        
        # Group interactions by action sequences
        action_groups = defaultdict(list)
        for interaction in interactions:
            if interaction.actions_taken:
                action_key = tuple(interaction.action_sequence)
                action_groups[action_key].append(interaction)
        
        # Find common query patterns for each action sequence
        for action_seq, group in action_groups.items():
            if len(group) >= self.min_frequency:
                # Extract query patterns
                queries = [i.user_query for i in group]
                common_phrases = self._extract_common_phrases(queries)
                
                if common_phrases:
                    pattern = InteractionPattern(
                        pattern_id=f"intent_{hash(action_seq)}",
                        pattern_type="intent",
                        query_patterns=common_phrases,
                        typical_actions=list(action_seq),
                        frequency=len(group),
                        success_rate=sum(1 for i in group if i.success) / len(group),
                        avg_execution_time=sum(i.execution_time or 0 for i in group) / len(group),
                        last_seen=max(i.timestamp for i in group),
                        confidence=min(len(group) / 10, 1.0)  # Confidence based on frequency
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_success_patterns(self, interactions: List[Interaction]) -> List[InteractionPattern]:
        """Detect patterns that consistently lead to successful outcomes."""
        patterns = []
        
        # Group successful interactions
        successful = [i for i in interactions if i.success]
        
        # Analyze common elements in successful interactions
        if len(successful) >= self.min_frequency:
            # Common query keywords in successful interactions
            all_keywords = []
            for interaction in successful:
                keywords = self._extract_keywords(interaction.user_query)
                all_keywords.extend(keywords)
            
            keyword_freq = Counter(all_keywords)
            common_keywords = [kw for kw, count in keyword_freq.most_common(10) 
                             if count >= self.min_frequency]
            
            if common_keywords:
                pattern = InteractionPattern(
                    pattern_id="success_keywords",
                    pattern_type="success",
                    query_patterns=common_keywords,
                    frequency=len(successful),
                    success_rate=1.0,  # By definition
                    confidence=0.8,
                    metadata={"keyword_frequencies": dict(keyword_freq.most_common(20))}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_correction_patterns(self, interactions: List[Interaction]) -> List[InteractionPattern]:
        """Detect patterns in user corrections to improve interpretation."""
        patterns = []
        
        # Find interactions with corrections
        corrections = [i for i in interactions if i.was_corrected]
        
        # Group corrections by similar misinterpretations
        misinterpretation_groups = defaultdict(list)
        for interaction in corrections:
            if interaction.ai_interpretation:
                # Simplified grouping by key terms
                key_terms = self._extract_keywords(interaction.ai_interpretation)
                key = tuple(sorted(key_terms)[:3])  # Use top 3 keywords as key
                misinterpretation_groups[key].append(interaction)
        
        # Create patterns from common corrections
        for key, group in misinterpretation_groups.items():
            if len(group) >= self.min_frequency:
                pattern = InteractionPattern(
                    pattern_id=f"correction_{hash(key)}",
                    pattern_type="correction",
                    query_patterns=[i.user_query for i in group],
                    typical_interpretation=group[0].ai_interpretation,  # Example misinterpretation
                    frequency=len(group),
                    success_rate=0.0,  # These led to corrections
                    confidence=0.9,  # High confidence these are problematic
                    metadata={
                        "common_corrections": [i.user_feedback for i in group if i.user_feedback],
                        "misunderstood_terms": list(key)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_workflow_patterns(self, interactions: List[Interaction]) -> List[InteractionPattern]:
        """Detect multi-step workflow patterns."""
        patterns = []
        
        # Group interactions by session and time proximity
        session_groups = defaultdict(list)
        for interaction in interactions:
            session_groups[interaction.session_id].append(interaction)
        
        # Analyze each session for workflow patterns
        for session_id, session_interactions in session_groups.items():
            # Sort by timestamp
            session_interactions.sort(key=lambda x: x.timestamp)
            
            # Find sequences of related interactions (within 10 minutes)
            workflows = []
            current_workflow = []
            last_time = None
            
            for interaction in session_interactions:
                if last_time and (interaction.timestamp - last_time) > timedelta(minutes=10):
                    if len(current_workflow) >= 2:  # At least 2 steps
                        workflows.append(current_workflow)
                    current_workflow = []
                
                current_workflow.append(interaction)
                last_time = interaction.timestamp
            
            if len(current_workflow) >= 2:
                workflows.append(current_workflow)
            
            # Extract patterns from workflows
            for workflow in workflows:
                # Create workflow signature
                steps = []
                for interaction in workflow:
                    step = f"{self._simplify_query(interaction.user_query)} -> {','.join(interaction.action_sequence)}"
                    steps.append(step)
                
                pattern = InteractionPattern(
                    pattern_id=f"workflow_{session_id}_{workflow[0].timestamp.timestamp()}",
                    pattern_type="workflow",
                    query_patterns=[i.user_query for i in workflow],
                    typical_actions=[action for i in workflow for action in i.action_sequence],
                    frequency=1,  # Single instance
                    success_rate=sum(1 for i in workflow if i.success) / len(workflow),
                    avg_execution_time=sum(i.execution_time or 0 for i in workflow),
                    last_seen=workflow[-1].timestamp,
                    confidence=0.6,
                    metadata={"workflow_steps": steps}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # Convert to lowercase and split
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stop words and short words
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        return keywords
    
    def _extract_common_phrases(self, queries: List[str]) -> List[str]:
        """Extract common phrases from a list of queries."""
        # Simple n-gram extraction
        all_ngrams = []
        
        for query in queries:
            words = self._extract_keywords(query)
            
            # Unigrams
            all_ngrams.extend(words)
            
            # Bigrams
            for i in range(len(words) - 1):
                all_ngrams.append(f"{words[i]} {words[i+1]}")
            
            # Trigrams
            for i in range(len(words) - 2):
                all_ngrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Count frequencies
        ngram_freq = Counter(all_ngrams)
        
        # Return common ones
        common = [ngram for ngram, count in ngram_freq.most_common(20) 
                 if count >= len(queries) * 0.3]  # Appears in 30% of queries
        
        return common
    
    def _simplify_query(self, query: str) -> str:
        """Simplify a query to its core intent."""
        keywords = self._extract_keywords(query)
        return ' '.join(keywords[:5])  # First 5 keywords
    
    def calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries (simple approach)."""
        # Extract keywords
        kw1 = set(self._extract_keywords(query1))
        kw2 = set(self._extract_keywords(query2))
        
        if not kw1 or not kw2:
            return 0.0
        
        # Jaccard similarity
        intersection = kw1 & kw2
        union = kw1 | kw2
        
        return len(intersection) / len(union)
    
    async def find_similar_interactions(self, query: str, limit: int = 5) -> List[Tuple[Interaction, float]]:
        """Find interactions with similar queries."""
        # Get recent interactions
        recent = await self.memory_manager.get_recent_interactions(limit=100)
        
        # Calculate similarities
        similarities = []
        for interaction_data in recent:
            if 'user_query' in interaction_data:
                interaction = Interaction.from_dict(interaction_data)
                similarity = self.calculate_query_similarity(query, interaction.user_query)
                if similarity > 0:
                    similarities.append((interaction, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:limit]