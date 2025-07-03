"""Core Sleep implementation."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
from enum import Enum

from ..core.memory import Memory
from ..core.pattern_detector import PatternDetector
from ..storage.hybrid_store import HybridStore
from .role_manager import RoleManager
from .roles import CommandRole
from .world_model_sleep import WorldModelSleep, GoalDirectedProposal

logger = logging.getLogger(__name__)

class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    async def analyze(self, prompt: str, **kwargs) -> str:
        """Analyze content and return insights."""
        ...
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content based on prompt."""
        ...

class AnalysisType(Enum):
    """Types of analysis the KBM can perform."""
    PATTERN_MINING = "pattern_mining"
    CONTEXT_OPTIMIZATION = "context_optimization"
    PROMPT_ENHANCEMENT = "prompt_enhancement"
    KNOWLEDGE_CONSOLIDATION = "knowledge_consolidation"
    ERROR_ANALYSIS = "error_analysis"

@dataclass
class Observation:
    """Single observation of main instance behavior."""
    timestamp: datetime
    interaction_type: str
    query: str
    context_used: Dict[str, Any]
    response: str
    tokens_used: int
    latency: float
    success: bool
    metadata: Dict[str, Any] = None

@dataclass
class Insight:
    """Insight derived from analysis."""
    type: AnalysisType
    confidence: float
    description: str
    recommendations: List[str]
    data: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class Sleep:
    """Sleep for background processing and knowledge base optimization through meta-cognitive analysis."""
    
    def __init__(
        self,
        memory: Memory,
        llm_providers: Dict[str, LLMProvider],
        config: Optional[Dict[str, Any]] = None,
        hybrid_store: Optional[HybridStore] = None
    ):
        self.memory = memory
        self.llms = llm_providers
        self.config = config or {}
        self.hybrid_store = hybrid_store
        
        # Configuration
        self.batch_size = self.config.get('batch_size', 10)
        self.analysis_interval = self.config.get('analysis_interval', 300)  # 5 minutes
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Sleep state
        self.observation_queue = asyncio.Queue()
        self.insights_cache = []
        self.is_running = False
        self._tasks = []
        
        # Role management with hybrid store for better discovery
        self.role_manager = RoleManager(memory, hybrid_store, config)
        self.current_role: Optional[CommandRole] = None
        
        # World model-based reasoning
        self.world_model_sleep = WorldModelSleep(
            memory, 
            llm_providers.get('analyzer') if llm_providers else None
        )
        
    async def start(self):
        """Start the sleep background tasks."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Initialize role manager
        await self.role_manager.initialize()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._observation_processor()),
            asyncio.create_task(self._periodic_analyzer()),
            asyncio.create_task(self._insight_applicator()),
            asyncio.create_task(self._role_monitor())
        ]
        
        logger.info("Sleep started with role management")
        
    async def stop(self):
        """Stop the sleep background tasks."""
        self.is_running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Sleep stopped")
        
    async def observe(self, observation: Observation):
        """Add an observation to the queue for processing."""
        # Add role context to observation
        if self.current_role:
            if observation.metadata is None:
                observation.metadata = {}
            observation.metadata['active_role'] = self.current_role.name
            observation.metadata['role_confidence'] = self.current_role.confidence_threshold
        
        await self.observation_queue.put(observation)
        
        # Immediate analysis for critical observations
        if self._is_critical(observation):
            await self._immediate_analysis(observation)
            
    def _is_critical(self, observation: Observation) -> bool:
        """Determine if an observation requires immediate analysis."""
        # High latency
        if observation.latency > 10.0:
            return True
            
        # Failure
        if not observation.success:
            return True
            
        # High token usage
        if observation.tokens_used > 10000:
            return True
            
        return False
        
    async def _immediate_analysis(self, observation: Observation):
        """Perform immediate analysis on critical observations."""
        try:
            # Analyze the specific issue
            if not observation.success:
                insight = await self._analyze_failure(observation)
            elif observation.latency > 10.0:
                insight = await self._analyze_performance(observation)
            else:
                insight = await self._analyze_resource_usage(observation)
                
            if insight and insight.confidence >= self.min_confidence:
                self.insights_cache.append(insight)
                await self._apply_insight(insight)
                
        except Exception as e:
            logger.error(f"Immediate analysis failed: {e}")
            
    async def _observation_processor(self):
        """Process observations from the queue."""
        batch = []
        
        while self.is_running:
            try:
                # Collect observations into batches
                timeout = 1.0 if batch else None
                observation = await asyncio.wait_for(
                    self.observation_queue.get(),
                    timeout=timeout
                )
                batch.append(observation)
                
                # Process batch when full or on timeout
                if len(batch) >= self.batch_size:
                    await self._process_observation_batch(batch)
                    batch = []
                    
            except asyncio.TimeoutError:
                # Process partial batch on timeout
                if batch:
                    await self._process_observation_batch(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Observation processing error: {e}")
                
    async def _process_observation_batch(self, observations: List[Observation]):
        """Process a batch of observations."""
        try:
            # Log to memory for pattern detection
            for obs in observations:
                await self.memory.log_interaction(
                    action=obs.interaction_type,
                    file_path=obs.metadata.get('file_path') if obs.metadata else None,
                    details={
                        'query': obs.query[:200],  # Truncate for storage
                        'tokens': obs.tokens_used,
                        'latency': obs.latency,
                        'success': obs.success
                    }
                )
                
            # Run pattern detection
            pattern_detector = PatternDetector(self.memory)
            new_patterns = await pattern_detector.check_for_patterns()
            
            if new_patterns:
                logger.info(f"Detected {len(new_patterns)} new patterns")
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            
    async def _periodic_analyzer(self):
        """Run periodic deep analysis."""
        while self.is_running:
            try:
                await asyncio.sleep(self.analysis_interval)
                
                # Get recent interactions
                recent = await self.memory.get_recent_context(limit=100)
                
                # Run various analyses
                insights = []
                
                # Pattern mining
                pattern_insights = await self._mine_patterns(recent)
                insights.extend(pattern_insights)
                
                # Context optimization
                context_insights = await self._optimize_context(recent)
                insights.extend(context_insights)
                
                # Knowledge consolidation
                consolidation_insights = await self._consolidate_knowledge()
                insights.extend(consolidation_insights)
                
                # Cache high-confidence insights
                for insight in insights:
                    if insight.confidence >= self.min_confidence:
                        self.insights_cache.append(insight)
                        
                logger.info(f"Periodic analysis generated {len(insights)} insights")
                
            except Exception as e:
                logger.error(f"Periodic analysis error: {e}")
                
    async def _mine_patterns(self, recent_context: Dict[str, Any]) -> List[Insight]:
        """Mine patterns from recent interactions."""
        if 'analyzer' not in self.llms:
            return []
            
        try:
            # Get patterns from memory
            patterns = await self.memory.get_all_patterns()
            
            # Prepare prompt for LLM analysis
            prompt = f"""Analyze the following patterns detected in the system:

Patterns:
{json.dumps(patterns[:10], indent=2, default=str)}

Recent Context:
{json.dumps(recent_context, indent=2, default=str)}

Provide insights about:
1. Which patterns are most valuable
2. Which patterns might be problematic
3. Opportunities for automation
4. Potential improvements

Return as JSON with structure:
{{
    "valuable_patterns": [...],
    "problematic_patterns": [...],
    "automation_opportunities": [...],
    "improvements": [...]
}}"""

            response = await self.llms['analyzer'].analyze(prompt)
            
            try:
                analysis = json.loads(response)
            except:
                analysis = {"raw_response": response}
                
            insights = []
            
            # Convert analysis to insights
            if 'valuable_patterns' in analysis:
                for pattern in analysis['valuable_patterns']:
                    insights.append(Insight(
                        type=AnalysisType.PATTERN_MINING,
                        confidence=0.8,
                        description=f"Valuable pattern identified: {pattern.get('name', 'Unknown')}",
                        recommendations=[pattern.get('recommendation', 'Continue using this pattern')],
                        data=pattern
                    ))
                    
            if 'automation_opportunities' in analysis:
                for opp in analysis['automation_opportunities']:
                    insights.append(Insight(
                        type=AnalysisType.PATTERN_MINING,
                        confidence=0.9,
                        description=f"Automation opportunity: {opp.get('description', 'Unknown')}",
                        recommendations=opp.get('steps', []),
                        data=opp
                    ))
                    
            return insights
            
        except Exception as e:
            logger.error(f"Pattern mining error: {e}")
            return []
            
    async def _optimize_context(self, recent_context: Dict[str, Any]) -> List[Insight]:
        """Optimize context usage based on recent interactions."""
        if 'optimizer' not in self.llms:
            return []
            
        try:
            # Analyze context effectiveness
            top_files = recent_context.get('top_files', [])
            
            prompt = f"""Analyze context usage patterns:

Most accessed files:
{json.dumps(top_files, indent=2)}

Total interactions: {recent_context.get('total_interactions', 0)}

Provide optimization suggestions:
1. Which files should always be in context?
2. Which files are accessed together?
3. What context is missing?
4. How to organize context better?

Format as JSON with confidence scores."""

            response = await self.llms['optimizer'].analyze(prompt)
            
            # Create context optimization insight
            insight = Insight(
                type=AnalysisType.CONTEXT_OPTIMIZATION,
                confidence=0.8,
                description="Context usage analysis",
                recommendations=[
                    f"Always include: {', '.join([f['file'] for f in top_files[:3]])}"
                ],
                data={'analysis': response}
            )
            
            return [insight]
            
        except Exception as e:
            logger.error(f"Context optimization error: {e}")
            return []
            
    async def _consolidate_knowledge(self) -> List[Insight]:
        """Consolidate and compress knowledge base."""
        try:
            # Get all patterns
            patterns = await self.memory.get_all_patterns()
            
            # Group similar patterns
            similar_groups = self._group_similar_patterns(patterns)
            
            insights = []
            for group in similar_groups:
                if len(group) > 1:
                    insight = Insight(
                        type=AnalysisType.KNOWLEDGE_CONSOLIDATION,
                        confidence=0.9,
                        description=f"Found {len(group)} similar patterns that can be merged",
                        recommendations=[
                            f"Merge patterns: {', '.join([p['name'] for p in group[:3]])}"
                        ],
                        data={'patterns': group}
                    )
                    insights.append(insight)
                    
            return insights
            
        except Exception as e:
            logger.error(f"Knowledge consolidation error: {e}")
            return []
            
    def _group_similar_patterns(self, patterns: List[Dict]) -> List[List[Dict]]:
        """Group patterns by similarity."""
        # Simple grouping by pattern type for now
        groups = {}
        for pattern in patterns:
            key = pattern.get('pattern_type', 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(pattern)
            
        return [group for group in groups.values() if len(group) > 1]
        
    async def _analyze_failure(self, observation: Observation) -> Optional[Insight]:
        """Analyze a failed interaction."""
        if 'analyzer' not in self.llms:
            return None
            
        try:
            prompt = f"""Analyze this failed interaction:

Query: {observation.query}
Context used: {json.dumps(observation.context_used, indent=2)}
Error/Response: {observation.response}

Identify:
1. Root cause of failure
2. Missing context or knowledge
3. How to prevent similar failures

Provide specific recommendations."""

            analysis = await self.llms['analyzer'].analyze(prompt)
            
            return Insight(
                type=AnalysisType.ERROR_ANALYSIS,
                confidence=0.85,
                description="Failure analysis",
                recommendations=[analysis],
                data={
                    'observation': observation.__dict__,
                    'analysis': analysis
                }
            )
            
        except Exception as e:
            logger.error(f"Failure analysis error: {e}")
            return None
            
    async def _analyze_performance(self, observation: Observation) -> Optional[Insight]:
        """Analyze performance issues."""
        return Insight(
            type=AnalysisType.CONTEXT_OPTIMIZATION,
            confidence=0.75,
            description=f"High latency detected: {observation.latency}s",
            recommendations=[
                "Reduce context size",
                "Pre-compute common queries",
                "Cache frequent responses"
            ],
            data={'latency': observation.latency}
        )
        
    async def _analyze_resource_usage(self, observation: Observation) -> Optional[Insight]:
        """Analyze resource usage."""
        return Insight(
            type=AnalysisType.CONTEXT_OPTIMIZATION,
            confidence=0.7,
            description=f"High token usage: {observation.tokens_used}",
            recommendations=[
                "Compress context",
                "Use more specific queries",
                "Remove redundant information"
            ],
            data={'tokens': observation.tokens_used}
        )
        
    async def _insight_applicator(self):
        """Apply insights to improve the system."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Process cached insights
                while self.insights_cache:
                    insight = self.insights_cache.pop(0)
                    await self._apply_insight(insight)
                    
            except Exception as e:
                logger.error(f"Insight application error: {e}")
                
    async def _apply_insight(self, insight: Insight):
        """Apply a single insight to improve the system."""
        try:
            if insight.type == AnalysisType.PATTERN_MINING:
                # Add discovered pattern to the pattern store
                pattern_data = insight.data
                if 'type' in pattern_data and 'description' in pattern_data:
                    await self.memory.store.add_pattern({
                        'name': f"Sleep: {pattern_data['description'][:50]}",
                        'description': pattern_data['description'],
                        'pattern_type': pattern_data['type'],
                        'sequence': [],  # Will be filled by pattern detector
                        'confidence': insight.confidence,
                        'source': 'sleep_analysis'
                    })
                    
            elif insight.type == AnalysisType.CONTEXT_OPTIMIZATION:
                # Store context optimization preferences
                for rec in insight.recommendations:
                    if rec.startswith("Always include:"):
                        files = rec.replace("Always include:", "").strip()
                        await self.memory.store.add_preference({
                            'key': 'always_include_files',
                            'value': files,
                            'category': 'context',
                            'confidence': insight.confidence,
                            'source': 'sleep'
                        })
                        
            elif insight.type == AnalysisType.ERROR_ANALYSIS:
                # Store error patterns for future prevention
                await self.memory.store.add_pattern({
                    'name': 'Error pattern',
                    'description': insight.description,
                    'pattern_type': 'error',
                    'sequence': [],
                    'confidence': insight.confidence,
                    'metadata': insight.data
                })
                
            logger.info(f"Applied insight: {insight.type.value}")
            
        except Exception as e:
            logger.error(f"Failed to apply insight: {e}")
            
    async def get_prompt_recommendation(self, query: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get prompt enhancement recommendations."""
        if 'enhancer' not in self.llms:
            return {'enhanced': query, 'reasoning': 'No enhancer LLM available'}
            
        try:
            # Get relevant patterns
            patterns = await self.memory.get_all_patterns()
            preferences = await self.memory.get_preferences()
            
            prompt = f"""Enhance this query for better results:

Original query: {query}
Current context files: {current_context.get('files', [])}
Known patterns: {len(patterns)}
Preferences: {json.dumps(preferences, indent=2)}

Provide:
1. Enhanced query with better specificity
2. Relevant context to include
3. Examples from similar past queries
4. Suggested output format

Return as JSON."""

            response = await self.llms['enhancer'].generate(prompt)
            
            try:
                enhancement = json.loads(response)
                return {
                    'original': query,
                    'enhanced': enhancement.get('query', query),
                    'context_suggestions': enhancement.get('context', []),
                    'examples': enhancement.get('examples', []),
                    'format': enhancement.get('format', ''),
                    'reasoning': enhancement.get('reasoning', '')
                }
            except json.JSONDecodeError:
                return {
                    'enhanced': query,
                    'reasoning': 'Failed to parse enhancement'
                }
                
        except Exception as e:
            logger.error(f"Prompt recommendation error: {e}")
            return {
                'enhanced': query,
                'reasoning': f'Error: {str(e)}'
            }
    
    async def _role_monitor(self):
        """Monitor and update active roles based on context."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Get current context
                recent_context = await self.memory.get_recent_context(limit=20)
                context = {
                    'task_description': '',
                    'files': [f['file'] for f in recent_context.get('top_files', [])],
                    'recent_actions': [i.get('action', '') for i in recent_context.get('recent_interactions', [])]
                }
                
                # Update active role
                new_role = await self.role_manager.get_active_role(context)
                if new_role != self.current_role:
                    self.current_role = new_role
                    logger.info(f"Active role changed to: {new_role.name if new_role else 'None'}")
                    
                    # Create insight about role change
                    if new_role:
                        insight = Insight(
                            type=AnalysisType.CONTEXT_OPTIMIZATION,
                            confidence=0.8,
                            description=f"Role activated: {new_role.name}",
                            recommendations=[new_role.description],
                            data={
                                'role': new_role.name,
                                'triggers': [t.pattern for t in new_role.context_triggers]
                            }
                        )
                        self.insights_cache.append(insight)
                
            except Exception as e:
                logger.error(f"Role monitoring error: {e}")
    
    async def get_role_based_prompt(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get prompt enhanced with role-specific context."""
        # Update role based on current context
        self.current_role = await self.role_manager.get_active_role(context)
        
        result = {
            'original': query,
            'enhanced': query,
            'role': None,
            'role_context': '',
            'reasoning': ''
        }
        
        if self.current_role:
            result['role'] = self.current_role.name
            result['role_context'] = self.current_role.to_prompt_context()
            
            # Enhance query with role-specific modifiers
            enhanced_parts = [query]
            
            # Add tool preferences hint
            if self.current_role.tool_preferences:
                top_tools = sorted(
                    self.current_role.tool_preferences.items(),
                    key=lambda x: -x[1]
                )[:3]
                if top_tools:
                    tool_hint = f"(Consider using: {', '.join([t[0] for t in top_tools])})"
                    enhanced_parts.append(tool_hint)
            
            result['enhanced'] = ' '.join(enhanced_parts)
            result['reasoning'] = f"Applied '{self.current_role.name}' role based on context"
            
            # Update role performance based on usage
            await self.role_manager.update_role_performance(self.current_role.name, True)
        
        return result
    
    async def get_role_statistics(self) -> Dict[str, Any]:
        """Get role usage statistics."""
        return self.role_manager.get_role_statistics()
    
    async def get_role_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get role recommendations for a given context."""
        return await self.role_manager.get_role_recommendations(context)
    
    async def get_world_model_proposals(self, recent_interactions: List[Dict[str, Any]]) -> List[GoalDirectedProposal]:
        """Get goal-directed proposals based on world model understanding."""
        try:
            # Analyze current state and goals
            world_state, inferred_goal = await self.world_model_sleep.analyze_current_state(
                recent_interactions
            )
            
            if not world_state:
                return []
            
            # Generate current state summary
            current_state = {
                "domain": world_state.domain,
                "entity_count": len(world_state.entities),
                "recent_success_rate": self._calculate_recent_success_rate(recent_interactions),
                "active_entities": list(world_state.entities.keys())[:10]
            }
            
            # Generate proposals
            proposals = await self.world_model_sleep.generate_proposals(current_state)
            
            logger.info(
                f"Generated {len(proposals)} world model proposals",
                extra={
                    "domain": world_state.domain,
                    "goal_type": inferred_goal.get("type") if inferred_goal else "unknown",
                    "goal_confidence": self.world_model_sleep.goal_confidence
                }
            )
            
            return proposals
            
        except Exception as e:
            logger.error(f"Failed to generate world model proposals: {e}")
            return []
    
    def _calculate_recent_success_rate(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate success rate from recent interactions."""
        if not interactions:
            return 1.0
        
        successes = sum(1 for i in interactions if i.get("success", False))
        return successes / len(interactions)
    
    async def update_world_model_from_feedback(self, proposal_id: str, feedback: str, success: bool):
        """Update world model based on feedback on proposals."""
        # Find the proposal (in a real system, we'd store these)
        # For now, just log the feedback
        logger.info(
            f"World model feedback received",
            extra={
                "proposal_id": proposal_id,
                "feedback": feedback[:100],
                "success": success
            }
        )
        
        # In future: Update world model confidence, refine goal understanding, etc.
    
    async def get_current_world_understanding(self) -> Dict[str, Any]:
        """Get the current world model understanding."""
        if not self.world_model_sleep.current_world:
            return {"status": "no_world_model"}
        
        world = self.world_model_sleep.current_world
        goal = self.world_model_sleep.inferred_goal
        
        return {
            "domain": world.domain,
            "entities": {
                "count": len(world.entities),
                "types": list(set(e.type.value for e in world.entities.values()))
            },
            "relations": {
                "count": len(world.relations),
                "types": list(set(r.type.value for r in world.relations))
            },
            "inferred_goal": {
                "type": goal.get("type") if goal else "unknown",
                "description": goal.get("goal_state") if goal else "not inferred",
                "confidence": self.world_model_sleep.goal_confidence
            },
            "metrics": world.metrics
        }