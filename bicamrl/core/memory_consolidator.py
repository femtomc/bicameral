"""Memory consolidation system for hierarchical memory management."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

from ..utils.logging_config import get_logger

logger = get_logger("memory_consolidator")


class MemoryConsolidator:
    """Manages memory consolidation from active to working to episodic to semantic."""
    
    def __init__(self, memory, llm_provider=None):
        self.memory = memory
        self.llm_provider = llm_provider
        
        # Count-based thresholds for memory transitions
        self.active_to_working_threshold = 10  # Consolidate after 10 interactions
        self.working_to_episodic_threshold = 5  # Consolidate after 5 working memories
        self.episodic_to_semantic_threshold = 10  # Extract semantic after 10 episodes
        
        # Consolidation parameters
        self.min_frequency_for_semantic = 5  # Min occurrences to become semantic
        self.similarity_threshold = 0.8  # For grouping similar memories
        self.min_interactions_per_session = 3  # Min interactions to form a work session
        
    async def consolidate_memories(self) -> Dict[str, int]:
        """Run memory consolidation process based on interaction counts."""
        stats = {
            "active_to_working": 0,
            "working_to_episodic": 0,
            "episodic_to_semantic": 0,
            "cleaned_up": 0
        }
        
        # Get all data
        all_interactions = await self.memory.store.get_recent_interactions(2000)
        all_patterns = await self.memory.get_all_patterns()
        
        # Separate unconsolidated interactions and existing memories
        unconsolidated_interactions = []
        consolidated_memories = {"working": [], "episodic": [], "semantic": []}
        
        # Track consolidated interaction IDs to avoid reprocessing
        consolidated_ids = set()
        
        # Categorize existing consolidated memories
        for pattern in all_patterns:
            pattern_type = pattern.get('pattern_type', '')
            if 'consolidated_' in pattern_type:
                memory_type = pattern_type.replace('consolidated_', '')
                if memory_type in consolidated_memories:
                    consolidated_memories[memory_type].append(pattern)
                    # Track IDs of interactions that were consolidated
                    metadata = pattern.get('metadata', {})
                    if 'source_interactions' in metadata:
                        consolidated_ids.update(metadata['source_interactions'])
        
        # Get unconsolidated interactions
        for interaction in all_interactions:
            # Use the actual interaction_id from the database
            interaction_id = interaction.get('interaction_id')
            if interaction_id and interaction_id not in consolidated_ids:
                unconsolidated_interactions.append(interaction)
        
        # Also check if we already have consolidated memories that haven't been tracked
        # This handles the case where consolidation runs multiple times
        existing_working_count = len(consolidated_memories["working"])
        existing_episodic_count = len(consolidated_memories["episodic"])
        
        # STEP 1: Active → Working Memory Consolidation
        # Group unconsolidated interactions by session
        sessions = defaultdict(list)
        for interaction in unconsolidated_interactions:
            session_id = interaction.get('session_id', 'default')
            sessions[session_id].append(interaction)
        
        # Consolidate sessions with enough interactions
        for session_id, session_interactions in sessions.items():
            if len(session_interactions) >= self.active_to_working_threshold:
                # Sort by timestamp
                session_interactions.sort(key=lambda x: x.get('timestamp', ''))
                
                # Take chunks of interactions for consolidation
                for i in range(0, len(session_interactions), self.active_to_working_threshold):
                    chunk = session_interactions[i:i + self.active_to_working_threshold]
                    if len(chunk) >= self.min_interactions_per_session:
                        summary = await self._create_work_summary(chunk)
                        if summary:
                            # Track source interactions using their actual IDs
                            source_ids = []
                            for i in chunk:
                                # Only use interactions that have interaction_id
                                if 'interaction_id' in i and i['interaction_id']:
                                    source_ids.append(i['interaction_id'])
                            
                            # Only store if we have tracked source IDs
                            if source_ids:
                                summary['source_interactions'] = source_ids
                                await self._store_consolidated_memory(summary, "working")
                                stats["active_to_working"] += 1
        
        # STEP 2: Working → Episodic Memory Consolidation
        # Check if we have enough working memories to consolidate
        working_memories = consolidated_memories["working"]
        
        # Filter out working memories that have already been consolidated to episodic
        unconsolidated_working = []
        for wm in working_memories:
            if not wm.get('metadata', {}).get('consolidated_to_episodic', False):
                unconsolidated_working.append(wm)
        
        logger.debug(f"Unconsolidated working memories: {len(unconsolidated_working)}, threshold: {self.working_to_episodic_threshold}")
        
        if len(unconsolidated_working) >= self.working_to_episodic_threshold:
            # Group working memories by related context (files, time proximity)
            working_groups = self._group_working_memories(unconsolidated_working)
            logger.debug(f"Working memory groups: {len(working_groups)}, sizes: {[len(g) for g in working_groups]}")
            
            for group in working_groups:
                if len(group) >= self.working_to_episodic_threshold:
                    episode = await self._create_episode_from_working(group)
                    if episode:
                        await self._store_consolidated_memory(episode, "episodic")
                        stats["working_to_episodic"] += 1
                        
                        # We can't directly mark them as consolidated in the pattern store,
                        # but we track them in the episode metadata
        
        # STEP 3: Episodic → Semantic Knowledge Extraction
        # Check patterns and episodic memories for semantic extraction
        episodic_memories = consolidated_memories["episodic"]
        
        # Extract from high-frequency patterns
        patterns = await self.memory.get_all_patterns()
        semantic_extracted = await self._extract_semantic_from_patterns(patterns)
        stats["episodic_to_semantic"] += semantic_extracted
        
        # Extract from episodic memories if we have enough
        if len(episodic_memories) >= self.episodic_to_semantic_threshold:
            semantic_extracted = await self._extract_semantic_from_episodes(episodic_memories)
            stats["episodic_to_semantic"] += semantic_extracted
        
        # Clean up consolidated memories that have been promoted
        stats["cleaned_up"] = await self._cleanup_promoted_memories()
        
        logger.info(f"Memory consolidation complete: {stats}")
        return stats
    
    def _group_working_memories(self, working_memories: List[Dict]) -> List[List[Dict]]:
        """Group working memories by similarity for episodic consolidation."""
        groups = []
        used = set()
        
        for i, memory in enumerate(working_memories):
            if i in used:
                continue
                
            group = [memory]
            used.add(i)
            
            # Find similar memories based on files and time proximity
            metadata1 = memory.get('metadata', {})
            files1 = set(metadata1.get('files_touched', []))
            
            for j, other in enumerate(working_memories[i+1:], i+1):
                if j in used:
                    continue
                    
                metadata2 = other.get('metadata', {})
                files2 = set(metadata2.get('files_touched', []))
                
                # Check file overlap
                if files1 and files2 and len(files1.intersection(files2)) > 0:
                    group.append(other)
                    used.add(j)
                elif not files1 and not files2:
                    # Group memories without files by session
                    if metadata1.get('session_id') == metadata2.get('session_id'):
                        group.append(other)
                        used.add(j)
            
            groups.append(group)
        
        return groups
    
    async def _create_episode_from_working(self, working_memories: List[Dict]) -> Optional[Dict]:
        """Create an episodic memory from multiple working memories."""
        if not working_memories:
            return None
            
        # Extract key information from working memories
        all_files = set()
        all_actions = []
        total_duration = 0
        sessions = set()
        
        for wm in working_memories:
            metadata = wm.get('metadata', {})
            all_files.update(metadata.get('files_touched', []))
            all_actions.extend(metadata.get('action_summary', {}).items())
            total_duration += metadata.get('duration_minutes', 0)
            sessions.add(metadata.get('session_id'))
        
        # Create episode summary
        episode = {
            "type": "episode",
            "name": f"Episode: {len(working_memories)} work sessions",
            "description": f"Consolidated episode from {len(working_memories)} work sessions",
            "files_involved": list(all_files),
            "total_duration_minutes": total_duration,
            "num_sessions": len(sessions),
            "num_working_memories": len(working_memories),
            "action_distribution": dict(all_actions),
            "source_working_memories": [wm.get('name', '') for wm in working_memories]
        }
        
        # Use LLM to create episode narrative if available
        if self.llm_provider:
            try:
                context = f"Episode summary from {len(working_memories)} work sessions:\\n"
                for wm in working_memories[:5]:  # Limit to avoid token overflow
                    metadata = wm.get('metadata', {})
                    context += f"- {metadata.get('type', 'work')}: {metadata.get('semantic_summary', 'No summary')}\\n"
                
                prompt = f"""Create an episode narrative that:
1. Describes the overall story of what was accomplished
2. Identifies the main theme or goal across these sessions
3. Highlights key achievements or milestones
4. Notes any challenges or patterns

{context}"""
                
                narrative = await self.llm_provider.complete(prompt, max_tokens=250)
                episode['episode_narrative'] = narrative
                episode['has_narrative'] = True
            except Exception as e:
                logger.warning(f"Failed to create episode narrative: {e}")
                episode['has_narrative'] = False
        
        return episode
    
    async def _extract_semantic_from_patterns(self, patterns: List[Dict]) -> int:
        """Extract semantic knowledge from high-frequency patterns."""
        extracted = 0
        
        for pattern in patterns:
            # Skip already consolidated patterns
            if 'consolidated_' in pattern.get('pattern_type', ''):
                continue
                
            # Check frequency threshold
            if pattern.get('frequency', 0) >= self.min_frequency_for_semantic:
                # Create semantic knowledge entry
                knowledge = await self._create_semantic_from_pattern(pattern)
                if knowledge:
                    await self._store_consolidated_memory(knowledge, "semantic")
                    extracted += 1
        
        return extracted
    
    async def _create_semantic_from_pattern(self, pattern: Dict) -> Optional[Dict]:
        """Create semantic knowledge from a pattern."""
        knowledge = {
            "type": "semantic_pattern",
            "name": f"Principle: {pattern['name']}",
            "description": f"Semantic principle derived from pattern '{pattern['name']}'",
            "frequency": pattern.get('frequency', 0),
            "confidence": pattern.get('confidence', 0.5),
            "pattern_type": pattern.get('pattern_type'),
            "key_attributes": {
                "original_pattern": pattern['name'],
                "sequence": pattern.get('sequence', [])
            }
        }
        
        # Use LLM to extract deeper principle if available
        if self.llm_provider:
            try:
                prompt = f"""Extract the general principle from this pattern:
Pattern: {pattern['name']}
Type: {pattern.get('pattern_type')}
Frequency: {pattern.get('frequency')} times
Description: {pattern.get('description')}

Provide:
1. The underlying principle or best practice
2. When to apply this principle
3. Why it's effective"""
                
                principle = await self.llm_provider.complete(prompt, max_tokens=200)
                knowledge['semantic_principle'] = principle
                knowledge['has_principle'] = True
            except Exception as e:
                logger.warning(f"Failed to extract semantic principle: {e}")
                knowledge['has_principle'] = False
        
        return knowledge
    
    async def _extract_semantic_from_episodes(self, episodes: List[Dict]) -> int:
        """Extract semantic knowledge from episodic memories."""
        if not episodes or not self.llm_provider:
            return 0
            
        # Analyze episodes for meta-patterns
        try:
            episode_summary = "Episode analysis:\\n"
            for ep in episodes[:10]:  # Limit to avoid token overflow
                metadata = ep.get('metadata', {})
                episode_summary += f"- {metadata.get('name', 'Episode')}: "
                episode_summary += f"{metadata.get('episode_narrative', metadata.get('description', 'No description'))}\\n"
            
            prompt = f"""Analyze these episodes to extract:
1. Common themes and patterns across episodes
2. General workflow principles
3. Best practices that emerge
4. Meta-level insights about the developer's approach

{episode_summary}"""
            
            meta_insights = await self.llm_provider.complete(prompt, max_tokens=300)
            
            knowledge = {
                "type": "semantic_meta",
                "name": "Meta-Insights from Episodes",
                "description": "High-level patterns extracted from episodic memories",
                "confidence": 0.8,
                "key_attributes": {
                    "num_episodes_analyzed": len(episodes),
                    "insights": meta_insights
                },
                "has_meta_analysis": True
            }
            
            await self._store_consolidated_memory(knowledge, "semantic")
            return 1
            
        except Exception as e:
            logger.warning(f"Failed to extract semantic from episodes: {e}")
            return 0
    
    async def _cleanup_promoted_memories(self) -> int:
        """Archive consolidated memories and clean up lower tiers."""
        total_archived = 0
        
        # Step 1: Archive raw interactions that are in working memories
        all_patterns = await self.memory.get_all_patterns()
        working_memories = [p for p in all_patterns if p.get('pattern_type') == 'consolidated_working']
        
        # Collect all interaction IDs that have been consolidated
        interactions_to_archive = set()
        for wm in working_memories:
            source_ids = wm.get('metadata', {}).get('source_interactions', [])
            interactions_to_archive.update(source_ids)
        
        # Archive interactions in batches
        if interactions_to_archive:
            batch_size = 100
            interaction_list = list(interactions_to_archive)
            
            for i in range(0, len(interaction_list), batch_size):
                batch = interaction_list[i:i + batch_size]
                try:
                    archived = await self.memory.store.archive_interactions(
                        interaction_ids=batch,
                        reason="consolidated_to_working"
                    )
                    total_archived += archived
                    logger.info(f"Archived {archived} interactions to working memory")
                except Exception as e:
                    logger.error(f"Failed to archive interactions: {e}")
        
        # Step 2: Archive working memories that are in episodes
        episodic_memories = [p for p in all_patterns if p.get('pattern_type') == 'consolidated_episodic']
        working_to_archive = []
        
        # Find which working memories have been consolidated to episodes
        for episode in episodic_memories:
            source_wm = episode.get('metadata', {}).get('source_working_memories', [])
            # Find the pattern IDs for these working memories
            for wm in working_memories:
                if wm.get('name') in source_wm or wm.get('metadata', {}).get('name') in source_wm:
                    working_to_archive.append(wm.get('id'))
        
        if working_to_archive:
            try:
                archived = await self.memory.store.archive_patterns(
                    pattern_ids=working_to_archive,
                    reason="consolidated_to_episodic"
                )
                total_archived += archived
                logger.info(f"Archived {archived} working memories to episodic")
            except Exception as e:
                logger.error(f"Failed to archive working memories: {e}")
        
        # Step 3: Optionally vacuum database after large cleanup
        if total_archived > 1000:
            try:
                await self.memory.store.vacuum_database()
                logger.info("Vacuumed database after large cleanup")
            except Exception as e:
                logger.warning(f"Failed to vacuum database: {e}")
        
        return total_archived
    
    
    async def _extract_semantic_knowledge(self, interactions: List[Dict]) -> int:
        """Extract semantic knowledge from patterns and repeated behaviors."""
        # Get all patterns
        patterns = await self.memory.get_all_patterns()
        
        semantic_knowledge = []
        
        # Extract knowledge from high-frequency patterns
        for pattern in patterns:
            if pattern.get('frequency', 0) >= self.min_frequency_for_semantic:
                knowledge = {
                    "type": "semantic_pattern",
                    "name": pattern['name'],
                    "description": f"Common pattern: {pattern['description']}",
                    "frequency": pattern['frequency'],
                    "confidence": pattern.get('confidence', 0.5),
                    "derived_from": pattern['pattern_type'],
                    "key_attributes": {
                        "sequence": pattern.get('sequence', []),
                        "context": pattern.get('context', {})
                    }
                }
                
                # Use LLM to extract deeper meaning
                if self.llm_provider:
                    try:
                        pattern_desc = f"""Pattern Analysis:
Name: {pattern['name']}
Type: {pattern['pattern_type']}
Frequency: {pattern['frequency']} occurrences
Sequence: {pattern.get('sequence', [])}
Description: {pattern['description']}
"""
                        
                        prompt = f"""Analyze this recurring pattern and extract:
1. The general principle or best practice it represents
2. When this pattern should be applied
3. Why this pattern is effective
4. Potential improvements or variations

{pattern_desc}
"""
                        
                        llm_response = await self.llm_provider.complete(prompt, max_tokens=200)
                        knowledge["semantic_principle"] = llm_response
                        knowledge["has_semantic_extraction"] = True
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract semantic principle: {e}")
                        knowledge["has_semantic_extraction"] = False
                
                semantic_knowledge.append(knowledge)
        
        # Extract knowledge from preferences
        preferences = await self.memory.get_preferences()
        for category, prefs in preferences.items():
            if prefs:
                knowledge = {
                    "type": "semantic_preference",
                    "name": f"Preferences: {category}",
                    "description": f"Learned preferences for {category}",
                    "confidence": 0.9,
                    "key_attributes": prefs
                }
                
                # Use LLM to synthesize preferences into principles
                if self.llm_provider and len(prefs) > 2:
                    try:
                        pref_desc = f"Category: {category}\nPreferences:\n"
                        for key, value in list(prefs.items())[:10]:  # Limit to avoid token overflow
                            pref_desc += f"- {key}: {value}\n"
                        
                        prompt = f"""Synthesize these preferences into:
1. Core principles for {category}
2. The underlying philosophy or approach
3. How these preferences work together
4. Recommendations for similar situations

{pref_desc}
"""
                        
                        llm_response = await self.llm_provider.complete(prompt, max_tokens=200)
                        knowledge["synthesized_principles"] = llm_response
                        knowledge["has_synthesis"] = True
                        
                    except Exception as e:
                        logger.warning(f"Failed to synthesize preferences: {e}")
                        knowledge["has_synthesis"] = False
                
                semantic_knowledge.append(knowledge)
        
        # Extract cross-cutting insights if we have LLM
        if self.llm_provider and len(patterns) > 5:
            try:
                # Create a summary of all patterns
                pattern_summary = "Detected patterns:\n"
                for p in patterns[:15]:  # Limit to top patterns
                    pattern_summary += f"- {p['name']} ({p['pattern_type']}): {p.get('frequency', 0)} times\n"
                
                prompt = f"""Analyze these patterns holistically and identify:
1. The developer's overall workflow style
2. Common themes across different patterns
3. Areas of strength and efficiency
4. Potential areas for improvement
5. Recommended best practices based on these patterns

{pattern_summary}
"""
                
                llm_response = await self.llm_provider.complete(prompt, max_tokens=300)
                
                meta_knowledge = {
                    "type": "semantic_meta",
                    "name": "Workflow Analysis",
                    "description": "Cross-cutting insights from all patterns",
                    "confidence": 0.8,
                    "key_attributes": {
                        "total_patterns": len(patterns),
                        "analysis": llm_response
                    },
                    "has_meta_analysis": True
                }
                semantic_knowledge.append(meta_knowledge)
                
            except Exception as e:
                logger.warning(f"Failed to extract meta insights: {e}")
        
        # Store semantic knowledge
        for knowledge in semantic_knowledge:
            await self._store_consolidated_memory(knowledge, "semantic")
            
        return len(semantic_knowledge)
    
    async def _create_work_summary(self, interactions: List[Dict]) -> Optional[Dict]:
        """Create a summary of a work session using LLM for semantic understanding."""
        if not interactions:
            return None
            
        # Extract basic information
        actions = [i['action'] for i in interactions]
        files = [i.get('file_path') for i in interactions if i.get('file_path')]
        details = [i.get('details', {}) for i in interactions]
        
        # Count action types
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action] += 1
            
        # Find the time range
        timestamps = [datetime.fromisoformat(i['timestamp']) for i in interactions]
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        # Basic summary
        basic_summary = {
            "type": "work_session",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration,
            "files_touched": list(set(files)),
            "action_summary": dict(action_counts),
            "total_actions": len(interactions),
            "session_id": interactions[0].get('session_id')
        }
        
        # Use LLM to extract semantic meaning if available
        if self.llm_provider:
            try:
                # Create a narrative of the work session
                narrative = f"Work session analysis:\n"
                narrative += f"Duration: {duration:.1f} minutes\n"
                narrative += f"Files worked on: {', '.join(set(files))}\n"
                narrative += f"Actions performed:\n"
                
                for i, interaction in enumerate(interactions[:10]):  # Limit to avoid token overflow
                    action = interaction['action']
                    file_path = interaction.get('file_path', 'N/A')
                    detail_str = str(interaction.get('details', {}))[:100]
                    narrative += f"  {i+1}. {action} on {file_path}: {detail_str}\n"
                
                # Ask LLM to summarize the work
                prompt = f"""Analyze this work session and provide:
1. A one-sentence summary of what was accomplished
2. The primary goal or task being worked on
3. Any patterns or workflow detected
4. Suggested improvements or next steps

{narrative}
"""
                
                llm_response = await self.llm_provider.complete(prompt, max_tokens=200)
                
                # Add semantic understanding to summary
                basic_summary["semantic_summary"] = llm_response
                basic_summary["has_semantic_analysis"] = True
                
            except Exception as e:
                logger.warning(f"Failed to get LLM summary: {e}")
                basic_summary["has_semantic_analysis"] = False
        else:
            basic_summary["has_semantic_analysis"] = False
        
        return basic_summary
    
    async def _create_episode_summary(self, interactions: List[Dict]) -> Optional[Dict]:
        """Create a summary of an episode with semantic understanding."""
        if not interactions:
            return None
            
        work_summary = await self._create_work_summary(interactions)
        if not work_summary:
            return None
            
        # Add episode-specific information
        work_summary["type"] = "episode"
        
        # Identify the main focus of the episode
        files = [i.get('file_path') for i in interactions if i.get('file_path')]
        if files:
            file_counts = defaultdict(int)
            for f in files:
                file_counts[f] += 1
            main_file = max(file_counts.items(), key=lambda x: x[1])[0]
            work_summary["main_focus"] = main_file
            
        # Identify the workflow pattern if any
        action_sequence = [i['action'] for i in interactions[:10]]  # First 10 actions
        work_summary["action_sequence_sample"] = action_sequence
        
        # Use LLM to extract episode narrative if available
        if self.llm_provider and work_summary.get("has_semantic_analysis"):
            try:
                # Create episode context
                episode_context = f"""Episode Context:
Total interactions: {len(interactions)}
Duration: {work_summary['duration_minutes']:.1f} minutes
Main file focus: {work_summary.get('main_focus', 'various files')}
Files touched: {', '.join(work_summary['files_touched'][:5])}
Previous work summary: {work_summary.get('semantic_summary', 'N/A')}

Action sequence pattern: {' -> '.join(action_sequence[:10])}
"""
                
                # Ask LLM to create episode narrative
                prompt = f"""Based on this episode of work, create:
1. A narrative description of what happened (2-3 sentences)
2. The likely problem being solved or feature being implemented
3. Key learnings or insights from this episode
4. Success indicators or blockers encountered

{episode_context}
"""
                
                llm_response = await self.llm_provider.complete(prompt, max_tokens=250)
                
                work_summary["episode_narrative"] = llm_response
                work_summary["has_episode_analysis"] = True
                
            except Exception as e:
                logger.warning(f"Failed to get episode narrative: {e}")
                work_summary["has_episode_analysis"] = False
        
        return work_summary
    
    async def _store_consolidated_memory(self, memory: Dict, memory_type: str) -> None:
        """Store a consolidated memory."""
        memory["memory_type"] = memory_type
        memory["consolidated_at"] = datetime.now().isoformat()
        
        # Store as a special type of pattern for now
        # In a full implementation, this would be a separate table
        await self.memory.store.add_pattern({
            "name": f"{memory_type}: {memory.get('name', memory['type'])}",
            "description": memory.get('description', str(memory)),
            "pattern_type": f"consolidated_{memory_type}",
            "sequence": [],
            "frequency": 1,
            "confidence": memory.get('confidence', 0.8),
            "metadata": memory
        })
    
    async def _cleanup_old_memories(self) -> int:
        """Clean up memories that have been fully consolidated to higher tiers."""
        # In a full implementation, this would:
        # 1. Archive interactions that are in working memories
        # 2. Archive working memories that are in episodes  
        # 3. Archive episodes that contributed to semantic knowledge
        # 4. Keep only the highest tier representation
        
        # For now, return the count from _cleanup_promoted_memories
        return await self._cleanup_promoted_memories()
    
    async def get_relevant_memories(self, context: str, 
                                  memory_types: List[str] = None) -> List[Dict]:
        """Retrieve relevant memories based on context."""
        if memory_types is None:
            memory_types = ["working", "episodic", "semantic"]
            
        relevant_memories = []
        all_patterns = await self.memory.get_all_patterns()
        
        # Filter by memory type and search for relevance
        context_lower = context.lower()
        for pattern in all_patterns:
            pattern_type = pattern.get('pattern_type', '')
            
            # Check if it's a consolidated memory
            for mem_type in memory_types:
                if f"consolidated_{mem_type}" in pattern_type:
                    # Simple relevance check
                    metadata = pattern.get('metadata', {})
                    if self._is_relevant(metadata, context_lower):
                        relevant_memories.append({
                            "memory": metadata,
                            "type": mem_type,
                            "relevance": self._calculate_relevance(metadata, context_lower)
                        })
                        
        # Sort by relevance
        relevant_memories.sort(key=lambda x: x['relevance'], reverse=True)
        return relevant_memories[:10]  # Return top 10
    
    def _is_relevant(self, memory: Dict, context: str) -> bool:
        """Check if a memory is relevant to the context."""
        # Simple keyword matching for now
        memory_str = str(memory).lower()
        context_words = context.split()
        
        matches = sum(1 for word in context_words if word in memory_str)
        return matches >= 2  # At least 2 word matches
    
    def _calculate_relevance(self, memory: Dict, context: str) -> float:
        """Calculate relevance score."""
        memory_str = str(memory).lower()
        context_words = context.split()
        
        matches = sum(1 for word in context_words if word in memory_str)
        return matches / len(context_words) if context_words else 0.0