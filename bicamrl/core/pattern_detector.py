"""Pattern detection and analysis."""

import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

from ..utils.logging_config import get_logger
from ..utils.log_utils import async_log_context, log_pattern_operation

logger = get_logger("pattern_detector")

class PatternDetector:
    """Detects patterns in interaction sequences."""
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.min_frequency = 3  # Minimum occurrences to be considered a pattern
        self.confidence_threshold = 0.6
        self.recency_weight_days = 7  # Days for exponential decay of pattern weight
        self.logger = logger
        
        logger.info(
            "PatternDetector initialized",
            extra={
                'min_frequency': self.min_frequency,
                'confidence_threshold': self.confidence_threshold,
                'recency_weight_days': self.recency_weight_days
            }
        )
        
    @log_pattern_operation("check")
    async def check_for_patterns(self) -> List[Dict[str, Any]]:
        """Check recent interactions for new patterns."""
        start_time = time.time()
        
        # Get recent interactions
        async with async_log_context(logger, "fetch_interactions", limit=100):
            context = await self.memory_manager.get_recent_context(limit=100)
            interactions = await self.memory_manager.store.get_recent_interactions(100)
        
        logger.info(
            f"Checking {len(interactions)} interactions for patterns",
            extra={
                'interaction_count': len(interactions),
                'session_id': context.get('session_id')
            }
        )
        
        # Detect different types of patterns
        new_patterns = []
        
        # File access patterns
        async with async_log_context(logger, "detect_file_patterns"):
            file_patterns = await self._detect_file_patterns(interactions)
            new_patterns.extend(file_patterns)
            if file_patterns:
                logger.info(
                    f"Found {len(file_patterns)} file access patterns",
                    extra={'pattern_type': 'file', 'count': len(file_patterns)}
                )
        
        # Action sequence patterns
        async with async_log_context(logger, "detect_sequence_patterns"):
            sequence_patterns = await self._detect_sequence_patterns(interactions)
            new_patterns.extend(sequence_patterns)
            if sequence_patterns:
                logger.info(
                    f"Found {len(sequence_patterns)} action sequence patterns",
                    extra={'pattern_type': 'sequence', 'count': len(sequence_patterns)}
                )
        
        # Workflow patterns
        async with async_log_context(logger, "detect_workflow_patterns"):
            workflow_patterns = await self._detect_workflow_patterns(interactions)
            new_patterns.extend(workflow_patterns)
            if workflow_patterns:
                logger.info(
                    f"Found {len(workflow_patterns)} workflow patterns",
                    extra={'pattern_type': 'workflow', 'count': len(workflow_patterns)}
                )
        
        # Store new patterns
        if new_patterns:
            async with async_log_context(logger, "store_patterns", count=len(new_patterns)):
                for i, pattern in enumerate(new_patterns):
                    await self.memory_manager.store.add_pattern(pattern)
                    logger.debug(
                        f"Stored pattern: {pattern.get('name', 'unnamed')}",
                        extra={
                            'pattern_id': pattern.get('id'),
                            'pattern_type': pattern.get('pattern_type'),
                            'confidence': pattern.get('confidence', 0)
                        }
                    )
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Pattern detection complete",
            extra={
                'duration_ms': duration_ms,
                'total_patterns_found': len(new_patterns),
                'file_patterns': len(file_patterns),
                'sequence_patterns': len(sequence_patterns),
                'workflow_patterns': len(workflow_patterns)
            }
        )
            
        return new_patterns
    
    async def _detect_file_patterns(self, interactions: List[Dict]) -> List[Dict]:
        """Detect patterns in file access."""
        patterns = []
        
        # Group by action type
        file_actions = defaultdict(list)
        for interaction in interactions:
            if interaction.get('file_path'):
                action = interaction['action']
                file_path = interaction['file_path']
                file_actions[action].append(file_path)
        
        # Look for files frequently accessed together
        for action, files in file_actions.items():
            if len(files) < self.min_frequency:
                continue
                
            # Find co-occurrence patterns
            file_pairs = []
            for i in range(len(files) - 1):
                if files[i] != files[i + 1]:  # Different files accessed sequentially
                    file_pairs.append((files[i], files[i + 1]))
            
            if file_pairs:
                pair_counts = Counter(file_pairs)
                for (file1, file2), count in pair_counts.items():
                    if count >= self.min_frequency:
                        patterns.append({
                            "name": f"File pair: {file1} → {file2}",
                            "description": f"Files often accessed together during {action}",
                            "pattern_type": "file_access",
                            "sequence": [file1, file2],
                            "frequency": count,
                            "confidence": min(count / len(files), 1.0)
                        })
        
        return patterns
    
    async def _detect_sequence_patterns(self, interactions: List[Dict]) -> List[Dict]:
        """Detect patterns in action sequences."""
        patterns = []
        
        # Sort interactions by timestamp (oldest first)
        sorted_interactions = sorted(interactions, key=lambda x: x['timestamp'])
        
        # Extract action sequences with timestamps
        actions_with_time = [(i['action'], i['timestamp']) for i in sorted_interactions if i.get('action')]
        
        # Look for repeated sequences of different lengths
        for seq_len in range(2, 6):  # Sequences of 2 to 5 actions
            sequences = defaultdict(int)
            sequence_weights = defaultdict(float)
            sequence_timestamps = defaultdict(list)
            
            for i in range(len(actions_with_time) - seq_len + 1):
                seq = tuple(action for action, _ in actions_with_time[i:i + seq_len])
                sequences[seq] += 1
                
                # Use the most recent timestamp in the sequence
                most_recent = max(ts for _, ts in actions_with_time[i:i + seq_len])
                weight = self._calculate_recency_weight(most_recent)
                sequence_weights[seq] += weight
                sequence_timestamps[seq].append(most_recent)
            
            # Find frequent sequences
            for seq, count in sequences.items():
                if count >= self.min_frequency:
                    weighted_frequency = sequence_weights[seq]
                    base_confidence = min(count / (len(actions_with_time) - seq_len + 1), 1.0)
                    recency_boost = weighted_frequency / count
                    adjusted_confidence = min(base_confidence * (0.7 + 0.3 * recency_boost), 1.0)
                    
                    patterns.append({
                        "name": f"Action sequence: {' → '.join(seq)}",
                        "description": f"Common sequence of {len(seq)} actions",
                        "pattern_type": "action_sequence",
                        "sequence": list(seq),
                        "frequency": count,
                        "weighted_frequency": weighted_frequency,
                        "confidence": adjusted_confidence,
                        "last_seen": max(sequence_timestamps[seq])
                    })
        
        return patterns
    
    async def _detect_workflow_patterns(self, interactions: List[Dict]) -> List[Dict]:
        """Detect higher-level workflow patterns."""
        patterns = []
        
        # Sort interactions by timestamp (oldest first) since they come in DESC order
        sorted_interactions = sorted(interactions, key=lambda x: x['timestamp'])
        
        # Group interactions by time proximity (within 5 minutes)
        workflows = []
        current_workflow = []
        last_time = None
        
        for interaction in sorted_interactions:
            try:
                timestamp = datetime.fromisoformat(interaction['timestamp'])
                
                if last_time and (timestamp - last_time) > timedelta(minutes=5):
                    # New workflow
                    if len(current_workflow) >= 3:  # Minimum workflow size
                        workflows.append(current_workflow)
                    current_workflow = [interaction]
                else:
                    current_workflow.append(interaction)
                    
                last_time = timestamp
            except Exception as e:
                logger.warning(f"Failed to parse timestamp: {e}")
                continue
        
        if len(current_workflow) >= 3:
            workflows.append(current_workflow)
        
        logger.debug(f"Found {len(workflows)} workflows from {len(interactions)} interactions")
        
        # Analyze workflows for patterns
        workflow_signatures = []
        for workflow in workflows:
            # Create a signature based on action types and file types
            signature = []
            for interaction in workflow:
                action = interaction['action']
                file_path = interaction.get('file_path', '')
                if file_path and '.' in file_path:
                    file_ext = file_path.split('.')[-1]
                else:
                    file_ext = 'unknown'
                signature.append(f"{action}:{file_ext}")
            workflow_signatures.append(tuple(signature))
        
        # Find common workflow patterns with recency weighting
        workflow_counts = Counter(workflow_signatures)
        workflow_weights = defaultdict(float)
        workflow_timestamps = defaultdict(list)
        
        # Calculate weighted frequency based on recency
        for i, (signature, workflow) in enumerate(zip(workflow_signatures, workflows)):
            # Use the most recent timestamp in the workflow
            most_recent_timestamp = max(w['timestamp'] for w in workflow)
            weight = self._calculate_recency_weight(most_recent_timestamp)
            workflow_weights[signature] += weight
            workflow_timestamps[signature].append(most_recent_timestamp)
        
        for signature, count in workflow_counts.items():
            if count >= self.min_frequency:
                # Create readable description
                steps = []
                for step in signature:
                    action, file_type = step.split(':')
                    if file_type != 'unknown':
                        steps.append(f"{action} {file_type} file")
                    else:
                        steps.append(action)
                
                # Calculate confidence with recency weighting
                weighted_frequency = workflow_weights[signature]
                base_confidence = min(count / len(workflows), 1.0)
                # Boost confidence for recent patterns
                recency_boost = weighted_frequency / count  # Average weight
                adjusted_confidence = min(base_confidence * (0.7 + 0.3 * recency_boost), 1.0)
                
                patterns.append({
                    "name": f"Workflow: {steps[0]} to {steps[-1]}",
                    "description": f"Common workflow with {len(steps)} steps",
                    "pattern_type": "workflow",
                    "sequence": list(signature),
                    "steps": steps,
                    "frequency": count,
                    "weighted_frequency": weighted_frequency,
                    "confidence": adjusted_confidence,
                    "last_seen": max(workflow_timestamps[signature])
                })
        
        return patterns
    
    async def find_matching_patterns(self, action_sequence: List[str], 
                                   fuzzy_threshold: float = 0.7) -> List[Dict]:
        """Find patterns matching a given action sequence."""
        all_patterns = await self.memory_manager.get_all_patterns()
        matches = []
        
        for pattern in all_patterns:
            if pattern.get('pattern_type') == 'action_sequence':
                pattern_seq = pattern.get('sequence', [])
                
                # Check for exact match
                if action_sequence == pattern_seq:
                    matches.append({
                        "pattern": pattern,
                        "match_type": "exact",
                        "confidence": pattern.get('confidence', 0.5),
                        "similarity": 1.0
                    })
                # Check for subsequence match
                elif self._is_subsequence(action_sequence, pattern_seq):
                    matches.append({
                        "pattern": pattern,
                        "match_type": "subsequence",
                        "confidence": pattern.get('confidence', 0.5) * 0.8,
                        "similarity": 0.8
                    })
                else:
                    # Check for fuzzy match
                    fuzzy_matches = self._find_fuzzy_matches(action_sequence, pattern_seq, fuzzy_threshold)
                    if fuzzy_matches:
                        best_match = max(fuzzy_matches, key=lambda x: x[1])
                        position, similarity = best_match
                        matches.append({
                            "pattern": pattern,
                            "match_type": "fuzzy",
                            "confidence": pattern.get('confidence', 0.5) * similarity,
                            "similarity": similarity,
                            "position": position
                        })
        
        # Sort by combined score (confidence * similarity)
        matches.sort(key=lambda x: x['confidence'] * x.get('similarity', 1.0), reverse=True)
        return matches
    
    def _is_subsequence(self, seq1: List[str], seq2: List[str]) -> bool:
        """Check if seq2 is a subsequence of seq1."""
        if len(seq2) > len(seq1):
            return False
            
        for i in range(len(seq1) - len(seq2) + 1):
            if seq1[i:i + len(seq2)] == seq2:
                return True
        return False
    
    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences using edit distance."""
        if not seq1 or not seq2:
            return 0.0
            
        # Use Levenshtein distance for similarity
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Convert distance to similarity (0-1 scale)
        max_len = max(m, n)
        similarity = 1.0 - (dp[m][n] / max_len)
        return similarity
    
    def _find_fuzzy_matches(self, target_seq: List[str], pattern_seq: List[str], 
                           threshold: float = 0.7) -> List[Tuple[int, float]]:
        """Find fuzzy matches of pattern_seq in target_seq."""
        matches = []
        pattern_len = len(pattern_seq)
        
        if pattern_len > len(target_seq):
            return matches
            
        # Slide window over target sequence
        for i in range(len(target_seq) - pattern_len + 1):
            window = target_seq[i:i + pattern_len]
            similarity = self._sequence_similarity(window, pattern_seq)
            
            if similarity >= threshold:
                matches.append((i, similarity))
                
        return matches
    
    def _calculate_recency_weight(self, timestamp_str: str) -> float:
        """Calculate weight based on how recent an interaction is."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            now = datetime.now()
            days_ago = (now - timestamp).days
            
            # Exponential decay: weight = e^(-days_ago / recency_weight_days)
            # This gives weight 1.0 for today, ~0.61 for 7 days ago, ~0.37 for 14 days ago
            import math
            weight = math.exp(-days_ago / self.recency_weight_days)
            return max(0.1, weight)  # Minimum weight of 0.1
        except:
            return 0.5  # Default weight if timestamp parsing fails
    
    async def update_pattern_confidence(self, pattern_id: str, delta: float) -> None:
        """Update confidence score for a pattern based on feedback."""
        patterns = await self.memory_manager.get_all_patterns()
        
        for pattern in patterns:
            if pattern.get('id') == pattern_id:
                old_confidence = pattern.get('confidence', 0.5)
                new_confidence = max(0.0, min(1.0, old_confidence + delta))
                pattern['confidence'] = new_confidence
                
                # Update in storage
                await self.memory_manager.store.update_pattern_confidence(
                    pattern_id, new_confidence
                )
                break