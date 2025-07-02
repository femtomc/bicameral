#!/usr/bin/env python3
"""Bicamrl MCP Server - Following MCP Best Practices."""

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import EmbeddedResource, ImageContent, Resource, TextContent, Tool

from .core.feedback_processor import FeedbackProcessor
from .core.interaction_logger import InteractionLogger
from .core.interaction_model import Action, FeedbackType, Interaction
from .core.interaction_pattern_detector import InteractionPatternDetector
from .core.memory import Memory
from .core.pattern_detector import PatternDetector
from .sleep.config_validator import SleepConfigValidator
from .sleep.llm_providers import create_llm_providers
from .sleep.prompt_optimizer import PromptOptimizer
from .sleep.sleep import Observation, Sleep
from .storage.hybrid_store import HybridStore
from .storage.sqlite_store import SQLiteStore
from .utils.logging_config import get_logger, setup_logging, setup_production_logging
from .utils.log_utils import async_log_context, create_interaction_logger as create_log_context
from .utils.mcp_logging import mcp_tool_logger, mcp_resource_logger, with_interaction_logging, log_tool_metric

# Configure logging based on environment
if os.environ.get("BICAMRL_ENV") == "production":
    setup_production_logging()
else:
    setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    
logger = get_logger("server")

# Global instances (initialized on server start)
memory: Optional[Memory] = None
pattern_detector: Optional[PatternDetector] = None
feedback_processor: Optional[FeedbackProcessor] = None
interaction_logger: Optional[InteractionLogger] = None
interaction_pattern_detector: Optional[InteractionPatternDetector] = None
sleep_layer: Optional[Sleep] = None
prompt_optimizer: Optional[PromptOptimizer] = None
config: Dict[str, Any] = {}


async def initialize_server():
    """Initialize server components."""
    start_time = time.time()
    logger.info("Starting Bicamrl server initialization")
    
    global \
        memory, \
        pattern_detector, \
        feedback_processor, \
        interaction_logger, \
        interaction_pattern_detector, \
        sleep_layer, \
        prompt_optimizer, \
        config

    # Load configuration
    async with async_log_context(logger, "load_configuration"):
        config = load_config()
        logger.info(
            "Configuration loaded",
            extra={
                'has_sleep': config.get('sleep_layer', {}).get('enabled', False),
                'config_source': 'Mind.toml' if config else 'defaults'
            }
        )

    # Get database path
    db_path = os.environ.get("MEMORY_DB_PATH", ".bicamrl/memory")
    logger.info(f"Using database path: {db_path}")

    # Check if we should enable LLM embeddings
    llm_embeddings = None
    if config.get("sleep_layer", {}).get("enabled", False):
        async with async_log_context(logger, "initialize_embeddings"):
            try:
                # Validate Sleep configuration first
                sleep_layer_config = SleepConfigValidator.validate(config.get("sleep_layer", {}))
                
                # Create LLM providers from config
                llm_providers = create_llm_providers(sleep_layer_config)
                logger.debug(
                    f"Created {len(llm_providers)} LLM providers",
                    extra={'providers': list(llm_providers.keys())}
                )
                
                # Get embeddings from the first provider that supports them
                for provider_name, provider in llm_providers.items():
                    if hasattr(provider, 'get_embeddings'):
                        llm_embeddings = provider.get_embeddings()
                        logger.info(
                            f"Embeddings initialized",
                            extra={'provider': provider_name, 'type': type(provider).__name__}
                        )
                        break
            except Exception as e:
                logger.warning(
                    f"Could not initialize LLM embeddings",
                    extra={'error_type': type(e).__name__, 'error': str(e)}
                )
    
    # Initialize core components with optional embeddings
    async with async_log_context(logger, "initialize_core_components"):
        memory = Memory(db_path, llm_embeddings)
        pattern_detector = PatternDetector(memory)
        feedback_processor = FeedbackProcessor(memory)
        interaction_logger = InteractionLogger(memory)
        interaction_pattern_detector = InteractionPatternDetector(memory)
        
        logger.info(
            "Core components initialized",
            extra={
                'has_embeddings': llm_embeddings is not None,
                'session_id': memory.session_id
            }
        )

    # Initialize Sleep if configured
    if config.get("sleep_layer", {}).get("enabled", False) and 'sleep_layer_config' in locals():
        try:
            # Validate Sleep configuration
            sleep_layer_config = SleepConfigValidator.validate(config.get("sleep_layer", {}))

            # Create LLM providers from config
            llm_providers = create_llm_providers(sleep_layer_config)

            # Initialize Sleep - it will use memory's hybrid store if available
            sleep_layer = Sleep(memory, llm_providers, sleep_layer_config, memory.hybrid_store)

            # Initialize prompt optimizer
            prompt_optimizer = PromptOptimizer(memory)
            
            # Set LLM provider on memory for semantic consolidation
            # Use the first available provider
            for provider_name, provider in llm_providers.items():
                memory.set_llm_provider(provider)
                logger.info(f"Using {provider_name} for memory consolidation")
                break

            # Start Sleep
            await sleep_layer.start()
            logger.info("Knowledge Base Maintainer initialized and started")
        except ValueError as e:
            logger.error(f"Invalid Sleep configuration: {e}")
            sleep_layer = None
            prompt_optimizer = None
        except Exception as e:
            logger.error(f"Failed to initialize Sleep: {e}")
            sleep_layer = None
            prompt_optimizer = None

    init_duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "Bicamrl server initialization complete",
        extra={
            'duration_ms': init_duration_ms,
            'components': {
                'memory': memory is not None,
                'pattern_detector': pattern_detector is not None,
                'feedback_processor': feedback_processor is not None,
                'sleep_layer': sleep_layer is not None,
                'embeddings': llm_embeddings is not None
            }
        }
    )


async def cleanup_server():
    """Cleanup server components."""
    logger.info("Starting server cleanup")
    
    if sleep_layer:
        async with async_log_context(logger, "stop_sleep_layer"):
            await sleep_layer.stop()
            logger.info("Sleep layer stopped")
    
    if memory and hasattr(memory, 'consolidate_memories'):
        try:
            logger.info("Running final memory consolidation")
            stats = await memory.consolidate_memories()
            logger.info(
                "Final consolidation complete",
                extra={'consolidation_stats': stats}
            )
        except Exception as e:
            logger.error(
                "Error during final consolidation",
                extra={'error_type': type(e).__name__},
                exc_info=True
            )
    
    logger.info("Server cleanup complete")


def load_config() -> Dict[str, Any]:
    """Load configuration from Mind.toml or environment."""
    config = {}

    # Try to load from Mind.toml first
    mind_toml_paths = [
        os.path.expanduser("~/.bicamrl/Mind.toml"),
        ".bicamrl/Mind.toml",
        "Mind.toml",
    ]

    for path in mind_toml_paths:
        if os.path.exists(path):
            try:
                import tomli

                with open(path, "rb") as f:
                    toml_config = tomli.load(f)
                # Map 'sleep' to 'sleep_layer' for backwards compatibility
                if "sleep" in toml_config:
                    config["sleep_layer"] = toml_config["sleep"]
                # Copy other sections
                for key in ["logging", "memory", "interaction", "performance"]:
                    if key in toml_config:
                        config[key] = toml_config[key]
                logger.info(f"Loaded configuration from {path}")
                break
            except ImportError:
                logger.warning("tomli not installed, trying JSON fallback")
                break
            except Exception as e:
                logger.error(f"Failed to load config from {path}: {e}")

    # Fallback to old JSON config
    if not config:
        config_paths = [
            ".bicamrl/config.json",
            "bicameral_config.json",
            os.path.expanduser("~/.bicamrl/config.json"),
        ]

        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        config = json.load(f)
                    logger.info(f"Loaded configuration from {path} (legacy JSON)")
                    break
                except Exception as e:
                    logger.error(f"Failed to load config from {path}: {e}")

    # Override with environment variables
    if os.environ.get("SLEEP_LAYER_ENABLED", "").lower() == "true":
        config.setdefault("sleep_layer", {})["enabled"] = True

    # LLM provider configuration from environment
    if os.environ.get("OPENAI_API_KEY"):
        config.setdefault("sleep_layer", {}).setdefault("llm_providers", {}).setdefault(
            "openai", {}
        )
        config["sleep_layer"]["llm_providers"]["openai"]["api_key"] = os.environ["OPENAI_API_KEY"]

    if os.environ.get("ANTHROPIC_API_KEY"):
        config.setdefault("sleep_layer", {}).setdefault("llm_providers", {}).setdefault(
            "claude", {}
        )
        config["sleep_layer"]["llm_providers"]["claude"]["api_key"] = os.environ[
            "ANTHROPIC_API_KEY"
        ]

    # Expand environment variables in config
    config = expand_env_vars(config)

    return config


def expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in config."""
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        return os.environ.get(var_name, obj)
    else:
        return obj


# Server lifecycle management
@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Manage server lifecycle."""
    await initialize_server()
    try:
        yield
    finally:
        await cleanup_server()


# Initialize FastMCP server with metadata
mcp = FastMCP(
    "bicamrl", dependencies=["aiosqlite", "pydantic", "aiohttp"], lifespan=server_lifespan
)


# Resources
@mcp.resource("memory://patterns")
@mcp_resource_logger("get_patterns")
async def get_patterns() -> Resource:
    """Get all learned patterns."""
    patterns = await memory.get_all_patterns()
    return Resource(
        uri="memory://patterns",
        name="Learned Patterns",
        description="All learned patterns from interactions",
        mimeType="application/json",
        text=json.dumps(patterns, indent=2),
    )


@mcp.resource("memory://patterns/workflows")
@mcp_resource_logger("get_workflow_patterns")
async def get_workflow_patterns() -> Resource:
    """Get workflow-specific patterns."""
    workflows = await memory.get_workflow_patterns()
    return Resource(
        uri="memory://patterns/workflows",
        name="Workflow Patterns",
        description="Detected workflow patterns",
        mimeType="application/json",
        text=json.dumps(workflows, indent=2),
    )


@mcp.resource("memory://preferences")
@mcp_resource_logger("get_preferences")
async def get_preferences() -> Resource:
    """Get developer preferences."""
    preferences = await memory.get_preferences()
    return Resource(
        uri="memory://preferences",
        name="Developer Preferences",
        description="Stored developer preferences and coding style",
        mimeType="application/json",
        text=json.dumps(preferences, indent=2),
    )


@mcp.resource("memory://context/recent")
@mcp_resource_logger("get_recent_context")
async def get_recent_context() -> Resource:
    """Get recent context."""
    context = await memory.get_recent_context()
    return Resource(
        uri="memory://context/recent",
        name="Recent Context",
        description="Recently accessed files and actions",
        mimeType="application/json",
        text=json.dumps(context, indent=2),
    )


@mcp.resource("memory://sleep/insights")
async def get_sleep_insights() -> Resource:
    """Get Sleep insights."""
    if not sleep_layer:
        return Resource(
            uri="memory://sleep/insights",
            name="Sleep Insights",
            description="Knowledge Base Maintainer not enabled",
            mimeType="application/json",
            text=json.dumps({"error": "Sleep not enabled"}),
        )

    insights = [i.__dict__ for i in sleep_layer.insights_cache]
    return Resource(
        uri="memory://sleep/insights",
        name="Sleep Insights",
        description="Insights from the Knowledge Base Maintainer",
        mimeType="application/json",
        text=json.dumps(insights, indent=2, default=str),
    )


@mcp.resource("memory://sleep/prompt-templates")
async def get_prompt_templates() -> Resource:
    """Get optimized prompt templates."""
    if not prompt_optimizer:
        return Resource(
            uri="memory://sleep/prompt-templates",
            name="Prompt Templates",
            description="Prompt optimizer not enabled",
            mimeType="application/json",
            text=json.dumps({"error": "Prompt optimizer not enabled"}),
        )

    templates = {
        name: {
            "name": t.name,
            "description": t.description,
            "variables": t.variables,
            "success_rate": t.success_rate,
            "usage_count": t.usage_count,
        }
        for name, t in prompt_optimizer.templates.items()
    }
    return Resource(
        uri="memory://sleep/prompt-templates",
        name="Prompt Templates",
        description="Optimized prompt templates",
        mimeType="application/json",
        text=json.dumps(templates, indent=2),
    )


@mcp.resource("memory://sleep/status")
async def get_sleep_status() -> Resource:
    """Get Sleep operational status."""
    if not sleep_layer:
        status = {"enabled": False, "reason": "Sleep not configured or initialization failed"}
    else:
        status = {
            "enabled": True,
            "is_running": sleep_layer.is_running,
            "config": {
                "batch_size": sleep_layer.batch_size,
                "analysis_interval": sleep_layer.analysis_interval,
                "min_confidence": sleep_layer.min_confidence,
            },
            "statistics": {
                "insights_cached": len(sleep_layer.insights_cache),
                "observation_queue_size": sleep_layer.observation_queue.qsize(),
                "llm_providers": list(sleep_layer.llms.keys()),
            },
        }

    return Resource(
        uri="memory://sleep/status",
        name="Sleep Status",
        description="Knowledge Base Maintainer operational status",
        mimeType="application/json",
        text=json.dumps(status, indent=2),
    )


@mcp.resource("memory://sleep/config")
async def get_sleep_config() -> Resource:
    """Get Sleep configuration."""
    if not config.get("sleep_layer"):
        sleep_layer_config = SleepConfigValidator.generate_default_config()
    else:
        sleep_layer_config = config.get("sleep_layer", {})

    return Resource(
        uri="memory://sleep/config",
        name="Sleep Configuration",
        description="Current Sleep configuration",
        mimeType="application/json",
        text=json.dumps(sleep_layer_config, indent=2),
    )


@mcp.resource("memory://sleep/roles/active")
async def get_active_roles() -> Resource:
    """Get currently active roles."""
    if not sleep_layer:
        return Resource(
            uri="memory://sleep/roles/active",
            name="Active Roles",
            description="Sleep not enabled",
            mimeType="application/json",
            text=json.dumps({"error": "Sleep not enabled"}),
        )

    # Get all roles
    all_roles = {**sleep_layer.role_manager.active_roles, **sleep_layer.role_manager.custom_roles}

    # Format role data
    roles_data = []
    for role in all_roles.values():
        roles_data.append(
            {
                "name": role.name,
                "description": role.description,
                "confidence_threshold": role.confidence_threshold,
                "success_rate": role.success_rate,
                "usage_count": role.usage_count,
                "is_active": role == sleep_layer.current_role,
            }
        )

    return Resource(
        uri="memory://sleep/roles/active",
        name="Active Roles",
        description="All available command roles",
        mimeType="application/json",
        text=json.dumps(roles_data, indent=2),
    )


@mcp.resource("memory://sleep/roles/current")
async def get_current_role() -> Resource:
    """Get the currently active role."""
    if not sleep_layer:
        return Resource(
            uri="memory://sleep/roles/current",
            name="Current Role",
            description="Sleep not enabled",
            mimeType="application/json",
            text=json.dumps({"error": "Sleep not enabled"}),
        )

    if sleep_layer.current_role:
        role_data = {
            "name": sleep_layer.current_role.name,
            "description": sleep_layer.current_role.description,
            "context_triggers": [
                {"type": t.trigger_type.value, "pattern": t.pattern, "weight": t.weight}
                for t in sleep_layer.current_role.context_triggers
            ],
            "communication_style": sleep_layer.current_role.communication_profile.style.value,
            "decision_rules": [
                {"condition": r.condition, "action": r.action, "priority": r.priority}
                for r in sleep_layer.current_role.decision_rules
            ],
            "tool_preferences": sleep_layer.current_role.tool_preferences,
        }
    else:
        role_data = {"message": "No role currently active"}

    return Resource(
        uri="memory://sleep/roles/current",
        name="Current Role",
        description="The currently active command role",
        mimeType="application/json",
        text=json.dumps(role_data, indent=2),
    )


@mcp.resource("memory://sleep/roles/statistics")
async def get_role_statistics() -> Resource:
    """Get role usage statistics."""
    if not sleep_layer:
        return Resource(
            uri="memory://sleep/roles/statistics",
            name="Role Statistics",
            description="Sleep not enabled",
            mimeType="application/json",
            text=json.dumps({"error": "Sleep not enabled"}),
        )

    stats = await sleep_layer.get_role_statistics()

    return Resource(
        uri="memory://sleep/roles/statistics",
        name="Role Statistics",
        description="Statistics about role usage and performance",
        mimeType="application/json",
        text=json.dumps(stats, indent=2),
    )


# Tools - New Interaction-based approach
@mcp.tool()
@mcp_tool_logger("start_interaction")
@with_interaction_logging
async def start_interaction(
    user_query: str, context: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Start tracking a new user interaction.

    Args:
        user_query: The user's natural language request
        context: Optional context (current files, recent actions, etc.)

    Returns:
        Dictionary with interaction_id
    """
    # Start the interaction
    interaction_id = interaction_logger.start_interaction(user_query, context)
    
    # Create a logger context for this interaction
    interaction_log = create_log_context(interaction_id, memory.session_id)
    interaction_log.log_query(user_query)

    # Check for similar past queries to provide suggestions
    similar_queries = []
    if memory.hybrid_store:
        similar_queries = await memory.search_similar_queries(user_query, k=3)
        
        if similar_queries:
            log_tool_metric(
                "similar_queries_found",
                len(similar_queries),
                "start_interaction",
                interaction_id=interaction_id
            )

    suggestions = {
        "interaction_id": interaction_id,
        "similar_queries": [
            {
                "query": metadata.get("text", ""),
                "actions": metadata.get("actions", []),
                "success": metadata.get("success", False),
            }
            for _, _, metadata in similar_queries
        ],
    }

    return suggestions


@mcp.tool()
@mcp_tool_logger("log_ai_interpretation")
async def log_ai_interpretation(
    interaction_id: str,
    interpretation: str,
    planned_actions: List[str],
    confidence: float = 0.5,
    active_role: Optional[str] = None,
) -> str:
    """
    Log what the AI understood from the user query.

    Args:
        interaction_id: The interaction ID from start_interaction
        interpretation: What the AI understood
        planned_actions: List of actions AI plans to take
        confidence: AI's confidence in interpretation (0-1)
        active_role: Which behavioral role is active (if any)

    Returns:
        Confirmation message
    """
    if not interaction_logger.current_interaction:
        return "Error: No active interaction. Call start_interaction first."

    if interaction_logger.current_interaction.interaction_id != interaction_id:
        return "Error: Interaction ID mismatch."

    interaction_logger.log_interpretation(
        interpretation=interpretation,
        planned_actions=planned_actions,
        confidence=confidence,
        role=active_role,
    )
    
    # Log the interpretation details
    interaction_log = create_log_context(interaction_id, memory.session_id)
    interaction_log.info(
        f"AI interpretation: {interpretation[:100]}...",
        interpretation=interpretation,
        planned_actions=planned_actions
    )
    
    # Log metrics
    log_tool_metric(
        "interpretation_confidence",
        confidence,
        "log_ai_interpretation",
        interaction_id=interaction_id,
        actions_count=len(planned_actions)
    )
    
    return "AI interpretation logged."


@mcp.tool()
@mcp_tool_logger("log_action")
async def log_action(
    action_type: str, target: Optional[str] = None, details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Log an individual action within the current interaction.

    Args:
        action_type: Type of action (e.g., "read_file", "edit_file")
        target: Target of the action (e.g., file path)
        details: Additional action details

    Returns:
        Confirmation message
    """
    if not interaction_logger.current_interaction:
        return "Error: No active interaction. Call start_interaction first."
    
    interaction_id = interaction_logger.current_interaction.interaction_id
    interaction_log = create_log_context(interaction_id, memory.session_id)
    
    action = interaction_logger.log_action(action_type, target, details)
    
    # Log the action details
    interaction_log.log_action(action_type, target or "none", details or {})
    
    # Mark action as completed immediately for now
    # In future, could track async execution
    interaction_logger.complete_action(action, result="Success")
    
    return f"Action '{action_type}' logged."


@mcp.tool()
@mcp_tool_logger("complete_interaction")
async def complete_interaction(
    feedback: Optional[str] = None, success: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Complete the current interaction and analyze patterns.

    Args:
        feedback: Optional user feedback
        success: Whether the interaction was successful

    Returns:
        Summary of the completed interaction
    """
    if not interaction_logger.current_interaction:
        return {"error": "No active interaction to complete."}
    
    interaction_id = interaction_logger.current_interaction.interaction_id
    interaction_log = create_log_context(interaction_id, memory.session_id)

    # Infer feedback type if feedback is provided
    feedback_type = None
    if feedback:
        feedback_lower = feedback.lower()
        if any(word in feedback_lower for word in ["no", "wrong", "actually", "instead"]):
            feedback_type = FeedbackType.CORRECTION
        elif any(word in feedback_lower for word in ["thanks", "great", "perfect", "good"]):
            feedback_type = FeedbackType.APPROVAL
        elif any(word in feedback_lower for word in ["also", "and", "then", "next"]):
            feedback_type = FeedbackType.FOLLOWUP
    
    # Complete the interaction
    completed = interaction_logger.complete_interaction(
        feedback=feedback, feedback_type=feedback_type, success=success
    )

    if not completed:
        return {"error": "Failed to complete interaction."}
    
    # Log feedback if provided
    if feedback:
        interaction_log.log_feedback(
            feedback_type.value if feedback_type else "general",
            feedback
        )

    # Store the completed interaction (will use hybrid store if available)
    await memory.store_interaction(completed)

    # Detect patterns from this interaction
    patterns = await interaction_pattern_detector.detect_patterns([completed])
    
    if patterns:
        log_tool_metric(
            "patterns_detected",
            len(patterns),
            "complete_interaction",
            interaction_id=interaction_id
        )

    # If Sleep is enabled, observe the completed interaction
    if sleep_layer:
        observation = Observation(
            timestamp=completed.timestamp,
            interaction_type="complete_interaction",
            query=completed.user_query,
            context_used=completed.query_context,
            response=str(completed.actions_taken),
            tokens_used=completed.tokens_used,
            latency=completed.execution_time or 0.0,
            success=completed.success,
            metadata=completed.to_dict(),
        )
        await sleep_layer.observe(observation)
    
    return {
        "interaction_id": completed.interaction_id,
        "success": completed.success,
        "actions_taken": len(completed.actions_taken),
        "execution_time": completed.execution_time,
        "patterns_detected": len(patterns),
        "feedback_type": completed.feedback_type.value if completed.feedback_type else "none",
    }


@mcp.tool()
@mcp_tool_logger("detect_pattern")
async def detect_pattern(action_sequence: List[str]) -> Dict[str, Any]:
    """
    Check if current action sequence matches known patterns.

    Args:
        action_sequence: Sequence of recent actions

    Returns:
        Matching patterns with confidence scores
    """
    matches = await pattern_detector.find_matching_patterns(action_sequence)
    
    # Log metrics for top matches
    if matches:
        log_tool_metric(
            "pattern_matches_found",
            len(matches),
            "detect_pattern",
            top_confidence=matches[0].get('confidence', 0) if matches else 0
        )
    
    return {"matches": matches}


@mcp.tool()
@mcp_tool_logger("get_relevant_context")
async def get_relevant_context(
    task_description: str, file_context: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get context relevant to current task.

    Args:
        task_description: Description of the current task
        file_context: Currently relevant files (optional)

    Returns:
        Relevant context including files, patterns, preferences, and similar past queries
    """
    # Get traditional context from memory manager
    context = await memory.get_relevant_context(
        task_description=task_description, file_context=file_context or []
    )

    # Search for similar past queries using vector search if available
    similar_queries = []
    if memory.hybrid_store:
        similar_queries = await memory.search_similar_queries(task_description, k=5)

    # Add similar query information to context
    high_similarity_queries = [
        {
            "query": metadata.get("text", ""),
            "actions": metadata.get("actions", []),
            "success": metadata.get("success", False),
            "timestamp": metadata.get("timestamp", ""),
        }
        for _, similarity, metadata in similar_queries
        if similarity > 0.7  # Only include highly similar queries
    ]
    context["similar_interactions"] = high_similarity_queries

    # Extract common patterns from similar successful interactions
    successful_similar = [m for _, _, m in similar_queries if m.get("success", False)]
    if successful_similar:
        common_actions = {}
        for metadata in successful_similar:
            for action in metadata.get("actions", []):
                common_actions[action] = common_actions.get(action, 0) + 1

        # Add most common successful actions
        suggested_actions = [
            action
            for action, count in sorted(common_actions.items(), key=lambda x: x[1], reverse=True)
            if count >= len(successful_similar) * 0.5  # Action appears in 50%+ of successful interactions
        ]
        context["suggested_actions"] = suggested_actions
        
        # Log metrics
        log_tool_metric(
            "suggested_actions_found",
            len(suggested_actions),
            "get_relevant_context",
            successful_similar_count=len(successful_similar)
        )
    
    return context


@mcp.tool()
@mcp_tool_logger("record_feedback")
async def record_feedback(
    feedback_type: str, message: str, interaction_id: Optional[str] = None
) -> str:
    """
    Record developer feedback.

    Args:
        feedback_type: Type of feedback (correct, prefer, pattern)
        message: Feedback message
        interaction_id: Optional interaction ID to associate feedback with

    Returns:
        Confirmation of feedback recording
    """
    if feedback_type not in ["correct", "prefer", "pattern"]:
        raise ValueError(f"Invalid feedback type: {feedback_type}")

    result = await feedback_processor.process_feedback(feedback_type=feedback_type, message=message)

    # If there's a current interaction, complete it with this feedback
    if interaction_logger.current_interaction:
        current_interaction_id = interaction_logger.current_interaction.interaction_id
        
        # Map old feedback types to new FeedbackType enum
        feedback_type_map = {
            "correct": FeedbackType.CORRECTION,
            "prefer": FeedbackType.CLARIFICATION,
            "pattern": FeedbackType.APPROVAL,
        }

        completed = interaction_logger.complete_interaction(
            feedback=message,
            feedback_type=feedback_type_map.get(feedback_type, FeedbackType.NONE),
            success=feedback_type != "correct",
        )

        if completed:
            await memory.store_interaction(completed)

    # If Sleep is enabled, this is important feedback
    if sleep_layer:
        observation = Observation(
            timestamp=datetime.now(),
            interaction_type="feedback",
            query=message,
            context_used={"feedback_type": feedback_type},
            response=result,
            tokens_used=0,
            latency=0.0,
            success=True,
            metadata={"feedback_type": feedback_type, "interaction_id": interaction_id},
        )
        await sleep_layer.observe(observation)
    
    return f"Feedback recorded: {result}"


@mcp.tool()
@mcp_tool_logger("observe_interaction")
async def observe_interaction(
    interaction_type: str,
    query: str,
    response: str,
    context_used: Optional[Dict[str, Any]] = None,
    tokens_used: Optional[int] = None,
    latency: Optional[float] = None,
    success: Optional[bool] = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Report an interaction to the Knowledge Base Maintainer.

    Args:
        interaction_type: Type of interaction
        query: The query or prompt
        response: The response generated
        context_used: Context used for the interaction
        tokens_used: Number of tokens used
        latency: Response latency in seconds
        success: Whether the interaction was successful
        metadata: Additional metadata

    Returns:
        Confirmation message
    """
    if not sleep_layer:
        return "Sleep not enabled"

    observation = Observation(
        timestamp=datetime.now(),
        interaction_type=interaction_type,
        query=query,
        context_used=context_used or {},
        response=response,
        tokens_used=tokens_used or 0,
        latency=latency or 0.0,
        success=success,
        metadata=metadata or {},
    )
    await sleep_layer.observe(observation)
    return "Interaction observed by Sleep"


@mcp.tool()
@mcp_tool_logger("optimize_prompt")
async def optimize_prompt(
    prompt: str, task_type: Optional[str] = None, context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get an optimized version of a prompt.

    Args:
        prompt: Original prompt
        task_type: Type of task (code_generation, bug_fix, refactoring, explanation, testing)
        context: Current context

    Returns:
        Optimized prompt with confidence and reasoning
    """
    if not prompt_optimizer:
        return {"error": "Prompt optimizer not enabled", "original": prompt, "optimized": prompt}

    valid_task_types = ["code_generation", "bug_fix", "refactoring", "explanation", "testing"]
    if task_type and task_type not in valid_task_types:
        raise ValueError(f"Invalid task type. Must be one of: {valid_task_types}")

    # Include active role in optimization
    active_role = sleep_layer.current_role if sleep_layer else None

    optimized = await prompt_optimizer.optimize_prompt(
        prompt=prompt, task_type=task_type, context=context or {}, active_role=active_role
    )

    result = {
        "original": optimized.original,
        "optimized": optimized.optimized,
        "context_additions": optimized.context_additions,
        "confidence": optimized.confidence,
        "reasoning": optimized.reasoning,
    }

    # Add role information if available
    if active_role:
        result["active_role"] = active_role.name
        result["role_context"] = active_role.to_prompt_context()

    return result


@mcp.tool()
@mcp_tool_logger("get_sleep_recommendation")
async def get_sleep_recommendation(
    query: str, context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get recommendations from the Knowledge Base Maintainer.

    Args:
        query: What to get recommendations for
        context: Current context

    Returns:
        Sleep recommendations
    """
    if not sleep_layer:
        return {"error": "Sleep not enabled"}

    recommendation = await sleep_layer.get_prompt_recommendation(
        query=query, current_context=context or {}
    )
    return recommendation


@mcp.tool()
async def search_memory(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search the memory system.

    Args:
        query: Search query
        limit: Maximum number of results

    Returns:
        Search results
    """
    results = await memory.search(query)
    return results[:limit]


@mcp.tool()
async def get_memory_stats() -> Dict[str, Any]:
    """
    Get memory system statistics.

    Returns:
        Memory statistics
    """
    stats = await memory.get_stats()
    return stats


@mcp.tool()
@mcp_tool_logger("consolidate_memories")
async def consolidate_memories() -> Dict[str, Any]:
    """
    Run memory consolidation process.

    Consolidates memories from active -> working -> episodic -> semantic.

    Returns:
        Consolidation statistics
    """
    stats = await memory.consolidate_memories()
    return {"message": "Memory consolidation complete", "statistics": stats}


@mcp.tool()
@mcp_tool_logger("get_memory_insights")
async def get_memory_insights(context: str) -> Dict[str, Any]:
    """
    Get memory insights relevant to a context.

    Args:
        context: The context to search for

    Returns:
        Relevant memories and insights
    """
    insights = await memory.get_memory_insights(context)
    return insights


@mcp.tool()
@mcp_tool_logger("get_role_recommendations")
async def get_role_recommendations(
    task_description: str, file_context: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get role recommendations for a given task.

    Args:
        task_description: Description of the task
        file_context: Files involved in the task

    Returns:
        List of recommended roles with activation scores
    """
    if not sleep_layer:
        return [{"error": "Sleep not enabled"}]

    context = {
        "task_description": task_description,
        "files": file_context or [],
        "recent_actions": [],  # Will be filled from memory
    }

    # Get recent actions
    recent = await memory.get_recent_context(limit=10)
    context["recent_actions"] = [i.get("action", "") for i in recent.get("recent_interactions", [])]

    recommendations = await sleep_layer.get_role_recommendations(context)
    return recommendations


@mcp.tool()
@mcp_tool_logger("apply_role")
async def apply_role(role_name: str) -> Dict[str, str]:
    """
    Manually apply a specific role.

    Args:
        role_name: Name of the role to apply

    Returns:
        Confirmation message
    """
    if not sleep_layer:
        return {"error": "Sleep not enabled"}

    # Find the role
    all_roles = {**sleep_layer.role_manager.active_roles, **sleep_layer.role_manager.custom_roles}

    if role_name not in all_roles:
        return {"error": f"Role '{role_name}' not found"}

    # Set as current role
    sleep_layer.current_role = all_roles[role_name]

    # Log the manual role application
    await sleep_layer.role_manager.update_role_performance(role_name, True)

    return {
        "message": f"Applied role: {role_name}",
        "description": sleep_layer.current_role.description,
    }


@mcp.tool()
@mcp_tool_logger("discover_roles")
async def discover_roles() -> Dict[str, Any]:
    """
    Manually trigger role discovery from interaction patterns.

    Returns:
        Discovery results including new roles found
    """
    if not sleep_layer:
        return {"error": "Sleep not enabled"}

    if not sleep_layer.role_manager.use_interaction_discovery:
        return {"error": "Interaction-based role discovery not available (hybrid store required)"}

    # Run discovery
    initial_role_count = len(sleep_layer.role_manager.custom_roles)

    try:
        await sleep_layer.role_manager.discover_new_roles()

        # Get results
        new_role_count = len(sleep_layer.role_manager.custom_roles) - initial_role_count
        all_roles = sleep_layer.role_manager.custom_roles

        # Format role information
        role_info = []
        for name, role in all_roles.items():
            role_info.append(
                {
                    "name": role.name,
                    "description": role.description,
                    "triggers": len(role.context_triggers),
                    "confidence": role.confidence_threshold,
                    "success_rate": role.success_rate,
                }
            )

        return {
            "new_roles_discovered": new_role_count,
            "total_custom_roles": len(all_roles),
            "roles": role_info,
            "discovery_method": "interaction-based"
            if sleep_layer.role_manager.use_interaction_discovery
            else "action-based",
        }

    except Exception as e:
        return {
            "error": f"Role discovery failed: {str(e)}",
            "hint": "Ensure you have sufficient interaction history (20+ interactions)",
        }
