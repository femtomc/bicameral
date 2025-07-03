"""LLM-based pattern detection and analysis."""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..utils.logging_config import get_logger
from ..utils.log_utils import async_log_context, log_pattern_operation
from .llm_service import LLMService

logger = get_logger("pattern_detector")


class PatternDetector:
    """Detects patterns in interaction sequences using LLM intelligence."""
    
    def __init__(self, memory_manager, llm_service: LLMService):
        self.memory_manager = memory_manager
        self.llm_service = llm_service
        self.min_frequency = 3  # Minimum occurrences to be considered a pattern
        self.confidence_threshold = 0.6
        self.logger = logger
        
        logger.info(
            "LLM-based PatternDetector initialized",
            extra={
                'min_frequency': self.min_frequency,
                'confidence_threshold': self.confidence_threshold
            }
        )
        
    @log_pattern_operation("check")
    async def check_for_patterns(self) -> List[Dict[str, Any]]:
        """Check recent interactions for new patterns using LLM analysis."""
        start_time = time.time()
        
        # Get recent interactions
        async with async_log_context(logger, "fetch_interactions", limit=100):
            context = await self.memory_manager.get_recent_context(limit=100)
            interactions = await self.memory_manager.store.get_complete_interactions(limit=100)
        
        logger.info(
            f"Analyzing {len(interactions)} interactions for patterns",
            extra={
                'interaction_count': len(interactions),
                'session_id': context.get('session_id')
            }
        )
        
        if len(interactions) < self.min_frequency:
            return []
        
        # Use LLM to analyze patterns
        patterns = await self._analyze_patterns_with_llm(interactions)
        
        # Filter by confidence threshold
        confident_patterns = [
            p for p in patterns 
            if p.get('confidence', 0) >= self.confidence_threshold
        ]
        
        duration = time.time() - start_time
        logger.info(
            f"Pattern detection completed",
            extra={
                'patterns_found': len(confident_patterns),
                'duration_seconds': duration
            }
        )
        
        return confident_patterns
        
    async def _analyze_patterns_with_llm(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to discover patterns in interactions."""
        # Prepare interaction summaries
        summaries = []
        for i, interaction in enumerate(interactions[-50:]):  # Last 50 interactions
            data = interaction if isinstance(interaction, dict) else json.loads(interaction)
            summaries.append({
                "query": data.get('user_query', '')[:100],
                "actions": len(data.get('actions_taken', [])),
                "success": data.get('success', False),
                "timestamp": data.get('timestamp', '')
            })
            
        prompt = f"""Analyze these user interactions to discover recurring patterns, workflows, and behaviors.

Recent Interactions:
{json.dumps(summaries, indent=2)}

Identify patterns such as:
1. Recurring workflows or action sequences
2. Common problem-solving approaches
3. Preferred tools or methods
4. Time-based patterns (e.g., daily routines)
5. Error patterns and recovery strategies

For each pattern, provide:
- Pattern name and description
- Frequency (how often it occurs)
- Confidence (0.0-1.0)
- Trigger conditions
- Recommendations

Return JSON array of patterns."""

        response = await self.llm_service.analyze_patterns(interactions)
        
        if response.error:
            logger.error(f"LLM pattern analysis failed: {response.error}")
            return []
            
        # Process and validate patterns
        patterns = []
        for pattern_data in response.content.get('patterns', []):
            pattern = {
                "name": pattern_data.get('name', 'Unknown Pattern'),
                "description": pattern_data.get('description', ''),
                "pattern_type": pattern_data.get('type', 'workflow'),
                "frequency": pattern_data.get('frequency', 1),
                "confidence": pattern_data.get('confidence', 0.5),
                "trigger_conditions": pattern_data.get('triggers', []),
                "recommendations": pattern_data.get('recommendations', []),
                "metadata": {
                    "llm_discovered": True,
                    "discovery_timestamp": datetime.now().isoformat(),
                    "source_interaction_count": len(interactions)
                }
            }
            
            # Only include patterns that meet minimum frequency
            if pattern['frequency'] >= self.min_frequency:
                patterns.append(pattern)
                
        return patterns
        
    async def find_similar_patterns(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find patterns similar to a query using LLM understanding."""
        # Get all stored patterns
        all_patterns = await self.memory_manager.store.get_patterns()
        
        if not all_patterns:
            return []
            
        # Use LLM to find similar patterns
        prompt = f"""Given this user query: "{query}"

And these discovered patterns:
{json.dumps([{"name": p['name'], "description": p.get('description', '')} for p in all_patterns[:20]], indent=2)}

Which patterns are most relevant to the query? Consider:
1. Semantic similarity
2. Potential applicability
3. Problem-solving relevance

Return the indices of the most relevant patterns with relevance scores."""

        response = await self.llm_service._execute_request(
            self.llm_service._build_request(
                prompt=prompt,
                response_format='json',
                temperature=0.3
            )
        )
        
        if response.error:
            # Fallback to simple text matching
            return self._fallback_similarity_search(query, all_patterns, limit)
            
        # Get relevant patterns based on LLM response
        relevant_patterns = []
        for match in response.content.get('relevant_patterns', []):
            idx = match.get('index', -1)
            if 0 <= idx < len(all_patterns):
                pattern = all_patterns[idx].copy()
                pattern['relevance_score'] = match.get('score', 0.5)
                relevant_patterns.append(pattern)
                
        # Sort by relevance and limit
        relevant_patterns.sort(key=lambda p: p['relevance_score'], reverse=True)
        return relevant_patterns[:limit]
        
    def _fallback_similarity_search(self, query: str, patterns: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Simple text-based similarity search as fallback."""
        query_lower = query.lower()
        scored_patterns = []
        
        for pattern in patterns:
            score = 0.0
            name_lower = pattern.get('name', '').lower()
            desc_lower = pattern.get('description', '').lower()
            
            # Check for word matches
            query_words = set(query_lower.split())
            name_words = set(name_lower.split())
            desc_words = set(desc_lower.split())
            
            word_overlap = len(query_words & (name_words | desc_words))
            score = word_overlap / len(query_words) if query_words else 0
            
            if score > 0:
                pattern_copy = pattern.copy()
                pattern_copy['relevance_score'] = score
                scored_patterns.append(pattern_copy)
                
        # Sort by score and return top matches
        scored_patterns.sort(key=lambda p: p['relevance_score'], reverse=True)
        return scored_patterns[:limit]
        
    async def update_pattern_confidence(self, pattern_id: str, feedback: str) -> None:
        """Update pattern confidence based on user feedback using LLM interpretation."""
        # Get the pattern
        patterns = await self.memory_manager.store.get_patterns()
        pattern = next((p for p in patterns if p.get('id') == pattern_id), None)
        
        if not pattern:
            logger.warning(f"Pattern {pattern_id} not found")
            return
            
        # Use LLM to interpret feedback and adjust confidence
        prompt = f"""Given this pattern:
Name: {pattern.get('name')}
Description: {pattern.get('description', 'No description')}
Current Confidence: {pattern.get('confidence', 0.5)}

And this user feedback: "{feedback}"

How should the confidence be adjusted? Consider:
- Positive feedback should increase confidence
- Negative feedback should decrease confidence
- Specific critiques might suggest pattern refinement

Return JSON with:
- new_confidence (0.0-1.0)
- reasoning
- suggested_refinements (if any)"""

        response = await self.llm_service._execute_request(
            self.llm_service._build_request(
                prompt=prompt,
                response_format='json',
                temperature=0.3
            )
        )
        
        if response.error:
            # Simple fallback adjustment
            if any(word in feedback.lower() for word in ['good', 'correct', 'helpful', 'yes']):
                new_confidence = min(0.95, pattern.get('confidence', 0.5) + 0.1)
            else:
                new_confidence = max(0.1, pattern.get('confidence', 0.5) - 0.1)
        else:
            new_confidence = response.content.get('new_confidence', pattern.get('confidence', 0.5))
            
        # Update pattern confidence
        await self.memory_manager.store.update_pattern_confidence(pattern_id, new_confidence)
        
        logger.info(
            f"Updated pattern confidence",
            extra={
                'pattern_id': pattern_id,
                'old_confidence': pattern.get('confidence', 0.5),
                'new_confidence': new_confidence,
                'feedback': feedback[:50]
            }
        )