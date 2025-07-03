"""Core memory management functionality."""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..storage.sqlite_store import SQLiteStore
from ..storage.hybrid_store import HybridStore
from ..utils.logging_config import get_logger
from ..utils.log_utils import (
    async_log_context,
    log_memory_operation,
    create_interaction_logger
)
from .memory_consolidator import MemoryConsolidator
from .pattern_detector import PatternDetector
from .llm_service import LLMService

logger = get_logger("memory")

class Memory:
    """Core memory system with hierarchical storage."""
    
    def __init__(self, db_path: str, llm_service: Optional[LLMService] = None, llm_embeddings=None, vector_backend: str = "basic"):
        start_time = time.time()
        self.logger = logger
        
        try:
            self.db_path = Path(db_path)
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(
                f"Initializing Memory system",
                extra={
                    'db_path': str(self.db_path),
                    'has_llm_service': llm_service is not None,
                    'has_embeddings': llm_embeddings is not None,
                    'vector_backend': vector_backend
                }
            )
            
            # Create log directory
            log_path = self.db_path.parent / "logs"
            log_path.mkdir(exist_ok=True)
            
            # Initialize LLM service (create default if not provided)
            if llm_service is None:
                llm_config = {
                    "default_provider": "openai",
                    "llm_providers": {}
                }
                self.llm_service = LLMService(llm_config)
            else:
                self.llm_service = llm_service
            
            # Initialize storage with timing
            store_start = time.time()
            self.store = SQLiteStore(self.db_path / "memory.db")
            logger.debug(
                "SQLite store initialized",
                extra={'duration_ms': (time.time() - store_start) * 1000}
            )
            
            # Initialize hybrid store if embeddings are provided
            if llm_embeddings:
                hybrid_start = time.time()
                self.hybrid_store = HybridStore(
                    self.db_path / "hybrid", 
                    llm_embeddings,
                    vector_backend=vector_backend
                )
                logger.debug(
                    "Hybrid store initialized",
                    extra={
                        'duration_ms': (time.time() - hybrid_start) * 1000,
                        'vector_backend': vector_backend
                    }
                )
                logger.info(
                    "Memory system initialized with hybrid storage",
                    extra={
                        'storage_type': 'hybrid', 
                        'path': str(self.db_path),
                        'vector_backend': vector_backend
                    }
                )
            else:
                self.hybrid_store = None
                logger.info(
                    "Memory system initialized with SQLite only",
                    extra={'storage_type': 'sqlite', 'path': str(self.db_path)}
                )
            
            self.session_id = datetime.now().isoformat()
            self.consolidator = MemoryConsolidator(self, self.llm_service)
            self.pattern_detector = PatternDetector(self, self.llm_service)
            
            init_time = (time.time() - start_time) * 1000
            logger.info(
                f"Memory system initialization complete",
                extra={
                    'duration_ms': init_time,
                    'session_id': self.session_id,
                    'storage_path': str(self.db_path)
                }
            )
            
        except Exception as e:
            logger.error(
                f"Failed to initialize Memory system: {e}",
                extra={'error_type': type(e).__name__},
                exc_info=True
            )
            raise
        
    def set_llm_provider(self, llm_provider):
        """Set LLM provider for semantic extraction during consolidation."""
        self.llm_provider = llm_provider
        self.consolidator.llm_provider = llm_provider
        logger.info(
            "LLM provider configured for semantic consolidation",
            extra={
                'provider_type': type(llm_provider).__name__,
                'session_id': self.session_id
            }
        )
        
    # New method to store interactions in both SQLite and hybrid store
    @log_memory_operation("active", "store")
    async def store_interaction(self, interaction):
        """Store interaction in both SQLite and hybrid store if available."""
        interaction_dict = interaction.to_dict() if hasattr(interaction, 'to_dict') else interaction
        interaction_id = interaction_dict.get('interaction_id', 'unknown')
        
        logger.debug(
            f"Storing interaction {interaction_id}",
            extra={
                'interaction_id': interaction_id,
                'session_id': self.session_id,
                'action': interaction_dict.get('action'),
                'has_hybrid': self.hybrid_store is not None
            }
        )
        
        # Store in SQLite with timing
        async with async_log_context(logger, "sqlite_store", interaction_id=interaction_id):
            await self.store.add_complete_interaction(interaction_dict)
        
        # Store in hybrid store if available
        if self.hybrid_store:
            async with async_log_context(logger, "hybrid_store", interaction_id=interaction_id):
                await self.hybrid_store.add_interaction(interaction)
                logger.info(
                    f"Interaction stored in hybrid store",
                    extra={'interaction_id': interaction_id, 'storage': 'hybrid'}
                )
        
    @log_memory_operation("pattern", "retrieve_all")
    async def get_all_patterns(self) -> List[Dict[str, Any]]:
        """Get all learned patterns."""
        patterns = await self.store.get_patterns()
        logger.info(
            f"Retrieved {len(patterns)} patterns",
            extra={'count': len(patterns), 'session_id': self.session_id}
        )
        return patterns
    
    async def get_workflow_patterns(self) -> List[Dict[str, Any]]:
        """Get workflow-specific patterns."""
        patterns = await self.store.get_patterns()
        return [p for p in patterns if p.get("pattern_type") == "workflow"]
    
    async def get_preferences(self) -> Dict[str, Dict[str, Any]]:
        """Get developer preferences organized by category."""
        prefs = await self.store.get_preferences()
        
        # Organize by category
        organized = {}
        for pref in prefs:
            category = pref.get("category", "general")
            if category not in organized:
                organized[category] = {}
            organized[category][pref["key"]] = pref["value"]
            
        return organized
    
    @log_memory_operation("context", "get_recent")
    async def get_recent_context(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent interaction context."""
        start_time = time.time()
        
        async with async_log_context(logger, "fetch_recent_interactions", limit=limit):
            interactions = await self.store.get_complete_interactions(limit=limit)
        
        logger.debug(
            f"Processing {len(interactions)} recent interactions",
            extra={'interaction_count': len(interactions), 'limit': limit}
        )
        
        # Extract file access patterns
        recent_files = []
        recent_actions = []
        
        for interaction in interactions:
            # Extract files and actions from the actions_taken field
            for action in interaction.get('actions_taken', []):
                if action.get('target') and '/' in str(action['target']):
                    recent_files.append(action['target'])
                recent_actions.append({
                    "action": action.get('action_type', 'unknown'),
                    "timestamp": interaction['timestamp'],
                    "file": action.get('target')
                })
        
        # Count file frequency
        file_counts = {}
        for f in recent_files:
            file_counts[f] = file_counts.get(f, 0) + 1
        
        # Sort by frequency
        top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        
        context = {
            "session_id": self.session_id,
            "recent_actions": recent_actions[:10],
            "top_files": [{"file": f, "count": c} for f, c in top_files[:5]],
            "total_interactions": len(interactions)
        }
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Recent context retrieved",
            extra={
                'duration_ms': duration_ms,
                'interaction_count': len(interactions),
                'unique_files': len(file_counts),
                'session_id': self.session_id
            }
        )
        
        return context
    
    async def get_relevant_context(self, task_description: str, 
                                 file_context: List[str]) -> Dict[str, Any]:
        """Get context relevant to a specific task using LLM understanding."""
        # Search for similar interactions
        similar = await self.search(task_description)
        
        # Use LLM-based pattern detector to find relevant patterns
        relevant_patterns = await self.pattern_detector.find_similar_patterns(
            task_description, 
            limit=5
        )
        
        # Get preferences that might apply
        all_prefs = await self.get_preferences()
        
        return {
            "task": task_description,
            "relevant_patterns": relevant_patterns,
            "similar_past_work": similar[:5],
            "applicable_preferences": all_prefs,
            "suggested_files": file_context
        }
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search through memory for relevant items."""
        results = []
        query_lower = query.lower()
        
        # Search patterns
        patterns = await self.store.get_patterns()
        for pattern in patterns:
            if query_lower in pattern.get('name', '').lower() or \
               query_lower in pattern.get('description', '').lower():
                results.append({
                    "type": "pattern",
                    "name": pattern['name'],
                    "description": pattern['description'],
                    "confidence": pattern['confidence']
                })
        
        # Search recent interactions
        interactions = await self.store.search_interactions_by_query(query, limit=100)
        for interaction in interactions:
            results.append({
                "type": "interaction",
                "name": f"Query: {interaction['user_query'][:50]}...",
                "description": f"Session: {interaction['session_id']}",
                "timestamp": interaction['timestamp'],
                "interaction_id": interaction['interaction_id']
            })
        
        # Search preferences
        preferences = await self.store.get_preferences()
        for pref in preferences:
            if query_lower in pref['key'].lower() or \
               query_lower in str(pref['value']).lower():
                results.append({
                    "type": "preference",
                    "name": pref['key'],
                    "description": str(pref['value']),
                    "category": pref.get('category', 'general')
                })
        
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        # Get complete interactions
        complete_interactions = await self.store.get_complete_interactions(limit=1000)
        patterns = await self.store.get_patterns()
        feedback = await self.store.get_feedback()
        
        # Calculate top files from complete interactions
        file_counts = {}
        recent_sessions = set()
        cutoff = datetime.now().timestamp() - 86400  # 24 hours
        
        # Process complete interactions
        for interaction in complete_interactions:
            # Count files from actions
            for action in interaction.get('actions_taken', []):
                if action.get('target') and '/' in str(action['target']):
                    f = action['target']
                    file_counts[f] = file_counts.get(f, 0) + 1
            
            # Count recent sessions
            try:
                ts = datetime.fromisoformat(interaction['timestamp']).timestamp()
                if ts > cutoff:
                    recent_sessions.add(interaction.get('session_id'))
            except:
                pass
        
        top_files = sorted(file_counts.keys(), 
                          key=lambda x: file_counts[x], 
                          reverse=True)[:10]
        
        return {
            "total_interactions": len(complete_interactions),
            "total_patterns": len(patterns),
            "total_feedback": len(feedback),
            "active_sessions": len(recent_sessions),
            "top_files": top_files
        }
    
    async def clear_specific(self, target: str) -> None:
        """Clear specific memory items."""
        if target == "patterns":
            await self.store.clear_patterns()
        elif target == "preferences":
            await self.store.clear_preferences()
        elif target == "feedback":
            await self.store.clear_feedback()
        elif target == "all":
            await self.store.clear_all()
        else:
            logger.warning(f"Unknown clear target: {target}")
    
    async def consolidate_memories(self) -> Dict[str, int]:
        """Run memory consolidation process."""
        return await self.consolidator.consolidate_memories()
    
    async def get_consolidated_memories(self, memory_types: Optional[List[str]] = None) -> List[Dict]:
        """Get consolidated memories by type."""
        if memory_types is None:
            memory_types = ["working", "episodic", "semantic"]
            
        all_patterns = await self.store.get_patterns()
        consolidated = []
        
        for pattern in all_patterns:
            pattern_type = pattern.get('pattern_type', '')
            for mem_type in memory_types:
                if f"consolidated_{mem_type}" in pattern_type:
                    consolidated.append({
                        "type": mem_type,
                        "data": pattern.get('metadata', {}),
                        "created": pattern.get('created_at'),
                        "confidence": pattern.get('confidence', 0.5)
                    })
                    
        return consolidated
    
    async def get_memory_insights(self, context: str) -> Dict[str, Any]:
        """Get insights from different memory levels relevant to context."""
        # Get relevant memories from consolidator
        relevant_memories = await self.consolidator.get_relevant_memories(context)
        
        # Get recent interactions
        recent = await self.get_recent_context(limit=10)
        
        # Get applicable patterns
        patterns = await self.get_all_patterns()
        relevant_patterns = []
        
        context_lower = context.lower()
        for pattern in patterns:
            if context_lower in pattern.get('name', '').lower() or \
               context_lower in pattern.get('description', '').lower():
                relevant_patterns.append(pattern)
                
        return {
            "consolidated_memories": relevant_memories[:5],
            "recent_context": recent,
            "relevant_patterns": relevant_patterns[:5],
            "memory_stats": await self.get_stats()
        }
    
    # Hybrid store methods (when available)
    async def search_similar_queries(self, query: str, k: int = 5, threshold: float = 0.7):
        """Search for similar queries using vector similarity."""
        if not self.hybrid_store:
            logger.warning("Hybrid store not available, falling back to keyword search")
            return await self.search(query)
        
        results = await self.hybrid_store.search_similar_queries(query, k, threshold)
        return [{"query": r[2]["text"], "similarity": r[1], "metadata": r[2]} for r in results]
    
    async def find_correction_patterns(self, limit: int = 20):
        """Find patterns where users corrected the AI."""
        if not self.hybrid_store:
            return []
        return await self.hybrid_store.find_correction_patterns(limit)
    
    async def find_successful_patterns(self, action_types: Optional[List[str]] = None, limit: int = 20):
        """Find patterns from successful interactions."""
        if not self.hybrid_store:
            return []
        return await self.hybrid_store.find_successful_patterns(action_types, limit)
    
    async def cluster_similar_queries(self, min_cluster_size: int = 3, similarity_threshold: float = 0.7):
        """Group similar queries into clusters."""
        if not self.hybrid_store:
            return []
        return await self.hybrid_store.cluster_similar_queries(min_cluster_size, similarity_threshold)
    
    async def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics showing counts by memory type."""
        all_patterns = await self.store.get_patterns()
        
        # Count patterns by memory type
        memory_counts = {
            "active": 0,
            "working": 0,
            "episodic": 0,
            "semantic": 0
        }
        
        # Count based on pattern_type field
        for pattern in all_patterns:
            pattern_type = pattern.get('pattern_type', '')
            if 'consolidated_working' in pattern_type:
                memory_counts['working'] += 1
            elif 'consolidated_episodic' in pattern_type:
                memory_counts['episodic'] += 1
            elif 'consolidated_semantic' in pattern_type:
                memory_counts['semantic'] += 1
            else:
                # Non-consolidated patterns are considered active
                memory_counts['active'] += 1
        
        # Also count complete interactions as active memories
        interactions = await self.store.get_complete_interactions(limit=1000)
        memory_counts['active'] += len(interactions)
        
        return memory_counts