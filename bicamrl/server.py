#!/usr/bin/env python3
"""Bicamrl MCP Server - Following MCP Best Practices."""

import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource

from .config_loader import get_memory_path, get_vector_backend
from .config_loader import load_config as load_json_config
from .core.feedback_processor import FeedbackProcessor
from .core.interaction_logger import InteractionLogger
from .core.interaction_model import FeedbackType
from .core.interaction_pattern_detector import InteractionPatternDetector
from .core.llm_service import LLMService
from .core.memory import Memory
from .core.pattern_detector import PatternDetector
from .health import health_checker
from .rate_limiter import RateLimiter
from .sleep.config_validator import SleepConfigValidator
from .sleep.llm_providers import create_llm_providers
from .sleep.prompt_optimizer import PromptOptimizer
from .sleep.sleep import Observation, Sleep
from .utils.log_utils import async_log_context
from .utils.log_utils import create_interaction_logger as create_log_context
from .utils.logging_config import get_logger, setup_logging, setup_production_logging
from .utils.mcp_logging import (
    log_tool_metric,
    mcp_resource_logger,
    mcp_tool_logger,
    with_interaction_logging,
)
from .utils.rate_limit_decorator import apply_rate_limiter_to_tools, rate_limited

# Don't configure logging at module import time!
# This will be done in initialize_server() instead
logger = get_logger("server")

# Global instances (initialized on server start)
memory: Optional[Memory] = None
pattern_detector: Optional[PatternDetector] = None
feedback_processor: Optional[FeedbackProcessor] = None
interaction_logger: Optional[InteractionLogger] = None
interaction_pattern_detector: Optional[InteractionPatternDetector] = None
sleep_layer: Optional[Sleep] = None
prompt_optimizer: Optional[PromptOptimizer] = None
rate_limiter: Optional[RateLimiter] = None
config: Dict[str, Any] = {}


async def initialize_server():
    """Initialize server components."""
    # Configure logging based on environment
    if os.environ.get("BICAMRL_ENV") == "production":
        setup_production_logging()
    else:
        setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

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
        rate_limiter, \
        config

    # Load configuration
    async with async_log_context(logger, "load_configuration"):
        config = load_config()
        logger.info(
            "Configuration loaded",
            extra={
                "has_sleep": config.get("sleep_layer", {}).get("enabled", False),
                "config_source": "Mind.toml" if config else "defaults",
            },
        )

    # Get database path and vector backend from config
    db_path = str(get_memory_path(config))
    vector_backend = get_vector_backend(config)
    logger.info(f"Using database path: {db_path}, vector backend: {vector_backend}")

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
                    extra={"providers": list(llm_providers.keys())},
                )

                # Get embeddings from the first provider that supports them
                for provider_name, provider in llm_providers.items():
                    if hasattr(provider, "get_embeddings"):
                        llm_embeddings = provider.get_embeddings()
                        logger.info(
                            "Embeddings initialized",
                            extra={"provider": provider_name, "type": type(provider).__name__},
                        )
                        break
            except Exception as e:
                logger.warning(
                    "Could not initialize LLM embeddings",
                    extra={"error_type": type(e).__name__, "error": str(e)},
                )

    # Initialize LLM service for all components
    llm_config = {
        "default_provider": config.get("default_llm_provider", "openai"),
        "llm_providers": config.get("llm_providers", {}),
        "rate_limit": config.get("llm_rate_limit", 60),
    }
    llm_service = LLMService(llm_config)

    # Initialize core components with LLM service
    async with async_log_context(logger, "initialize_core_components"):
        memory = Memory(
            db_path,
            llm_service=llm_service,
            llm_embeddings=llm_embeddings,
            vector_backend=vector_backend,
        )
        # pattern_detector is now part of memory
        feedback_processor = FeedbackProcessor(memory)
        interaction_logger = InteractionLogger(memory)
        interaction_pattern_detector = InteractionPatternDetector(memory)

        logger.info(
            "Core components initialized",
            extra={
                "has_llm_service": True,
                "has_embeddings": llm_embeddings is not None,
                "session_id": memory.session_id,
                "default_llm_provider": llm_config["default_provider"],
            },
        )

    # Initialize rate limiter
    async with async_log_context(logger, "initialize_rate_limiter"):
        rate_limiter_config = config.get("rate_limiting", {})
        rate_limiter = RateLimiter(
            requests_per_minute=rate_limiter_config.get("requests_per_minute", 60),
            burst_size=rate_limiter_config.get("burst_size", 10),
            window_seconds=rate_limiter_config.get("window_seconds", 60),
        )
        logger.info(
            "Rate limiter initialized",
            extra={
                "requests_per_minute": rate_limiter.requests_per_minute,
                "burst_size": rate_limiter.burst_size,
                "window_seconds": rate_limiter.window_seconds,
            },
        )

    # Initialize Sleep if configured
    if config.get("sleep_layer", {}).get("enabled", False) and "sleep_layer_config" in locals():
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
            "duration_ms": init_duration_ms,
            "components": {
                "memory": memory is not None,
                "pattern_detector": pattern_detector is not None,
                "feedback_processor": feedback_processor is not None,
                "sleep_layer": sleep_layer is not None,
                "embeddings": llm_embeddings is not None,
            },
        },
    )


async def cleanup_server():
    """Cleanup server components."""
    logger.info("Starting server cleanup")

    if sleep_layer:
        async with async_log_context(logger, "stop_sleep_layer"):
            await sleep_layer.stop()
            logger.info("Sleep layer stopped")

    if memory and hasattr(memory, "consolidate_memories"):
        try:
            logger.info("Running final memory consolidation")
            stats = await memory.consolidate_memories()
            logger.info("Final consolidation complete", extra={"consolidation_stats": stats})
        except Exception as e:
            logger.error(
                "Error during final consolidation",
                extra={"error_type": type(e).__name__},
                exc_info=True,
            )

    logger.info("Server cleanup complete")


def create_default_mind_toml():
    """Create default Mind.toml with Claude Code configuration."""
    default_config = """# Bicamrl Configuration - Auto-generated
# This configuration uses Claude Code for all Sleep agents

[sleep]
enabled = true
batch_size = 10
analysis_interval = 300  # seconds between deep analysis
min_confidence = 0.7
discovery_interval = 86400  # seconds between role discovery (24 hours)

# Claude Code configuration (no API key needed!)
[sleep.llm_providers.claude_code]
type = "claude_code"
enabled = true
temperature = 0.7

# Use Claude Code for all Sleep agents
[sleep.roles]
analyzer = "claude_code"      # Analyzes patterns and behaviors
generator = "claude_code"     # Generates insights and recommendations
enhancer = "claude_code"      # Enhances prompts
optimizer = "claude_code"     # Optimizes performance

[logging]
level = "INFO"
file = "~/.bicamrl/logs/bicamrl.log"

[memory]
consolidation_interval = 3600  # seconds (1 hour)
max_active_memories = 1000

[roles]
auto_discover = true
min_interactions_per_role = 10
max_active_roles = 10
"""

    # Create .bicamrl directory if it doesn't exist
    bicamrl_dir = Path.home() / ".bicamrl"
    bicamrl_dir.mkdir(exist_ok=True)

    # Create logs directory
    logs_dir = bicamrl_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Write Mind.toml
    mind_toml_path = bicamrl_dir / "Mind.toml"
    if not mind_toml_path.exists():
        mind_toml_path.write_text(default_config)
        logger.info(f"Created default Mind.toml at {mind_toml_path}")
        logger.info("Default configuration uses Claude Code for all Sleep agents")
        return True
    return False


def load_config() -> Dict[str, Any]:
    """Load configuration from Mind.toml and JSON config files."""
    # Create default Mind.toml if none exists
    create_default_mind_toml()

    # Start with JSON config (includes vector store settings)
    config = load_json_config()

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

    # If sleep is enabled but no providers configured, default to claude_code
    if config.get("sleep_layer", {}).get("enabled", False):
        if not config.get("sleep_layer", {}).get("llm_providers"):
            logger.info("No LLM providers configured, defaulting to Claude Code")
            config.setdefault("sleep_layer", {}).setdefault("llm_providers", {})
            config["sleep_layer"]["llm_providers"]["claude_code"] = {
                "type": "claude_code",
                "enabled": True,
                "temperature": 0.7,
            }
            # Also set default roles to use claude_code
            config["sleep_layer"]["roles"] = {
                "analyzer": "claude_code",
                "generator": "claude_code",
                "enhancer": "claude_code",
                "optimizer": "claude_code",
            }

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
    # Initialize server (including logging setup)
    await initialize_server()

    # Apply rate limiter to all tools
    if rate_limiter:
        apply_rate_limiter_to_tools(server, rate_limiter)
        logger.info("Rate limiting applied to MCP tools")

    try:
        yield
    finally:
        await cleanup_server()


# Initialize FastMCP server with metadata
mcp = FastMCP(
    "bicamrl", dependencies=["aiosqlite", "pydantic", "aiohttp"], lifespan=server_lifespan
)


# Resources


# Health Check Resources
@mcp.resource("health://status")
@mcp_resource_logger("health_status")
async def health_status() -> Resource:
    """Get overall health status of the server."""
    health_data = await health_checker.check_all(
        memory=memory,
        sleep_layer=sleep_layer,
        pattern_detector=pattern_detector,
        feedback_processor=feedback_processor,
    )

    return Resource(
        uri="health://status",
        name="Health Status",
        description="Overall health status of Bicamrl server",
        mimeType="application/json",
        text=json.dumps(health_data, indent=2, default=str),
    )


@mcp.resource("health://ready")
@mcp_resource_logger("health_ready")
async def health_ready() -> Resource:
    """Check if server is ready to handle requests."""
    is_ready = memory is not None and pattern_detector is not None

    ready_data = {
        "ready": is_ready,
        "timestamp": datetime.now().isoformat(),
        "components": {
            "memory": memory is not None,
            "pattern_detector": pattern_detector is not None,
            "feedback_processor": feedback_processor is not None,
            "interaction_logger": interaction_logger is not None,
        },
    }

    return Resource(
        uri="health://ready",
        name="Readiness Check",
        description="Check if server is ready to handle requests",
        mimeType="application/json",
        text=json.dumps(ready_data, indent=2),
    )


@mcp.resource("health://live")
@mcp_resource_logger("health_live")
async def health_live() -> Resource:
    """Simple liveness check."""
    return Resource(
        uri="health://live",
        name="Liveness Check",
        description="Simple liveness probe",
        mimeType="application/json",
        text=json.dumps(
            {
                "alive": True,
                "timestamp": datetime.now().isoformat(),
                "server": "bicamrl",
                "version": "1.0.0",
            },
            indent=2,
        ),
    )


@mcp.resource("health://rate-limits")
@mcp_resource_logger("rate_limit_status")
async def rate_limit_status() -> Resource:
    """Get rate limiting status and metrics."""
    if not rate_limiter:
        return Resource(
            uri="health://rate-limits",
            name="Rate Limit Status",
            description="Rate limiting not configured",
            mimeType="application/json",
            text=json.dumps({"error": "Rate limiting not configured"}, indent=2),
        )

    metrics = rate_limiter.get_metrics()

    return Resource(
        uri="health://rate-limits",
        name="Rate Limit Status",
        description="Current rate limiting metrics and configuration",
        mimeType="application/json",
        text=json.dumps(metrics, indent=2, default=str),
    )


# Pattern Resources
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


@mcp.resource("memory://sleep/world-model")
async def get_world_model_resource() -> Resource:
    """Get current world model understanding."""
    if not sleep_layer:
        return Resource(
            uri="memory://sleep/world-model",
            name="World Model",
            description="Sleep layer not enabled",
            mimeType="application/json",
            text=json.dumps({"error": "Sleep layer not enabled"}),
        )

    try:
        world_understanding = await sleep_layer.get_current_world_understanding()
        return Resource(
            uri="memory://sleep/world-model",
            name="World Model",
            description="Current world model understanding",
            mimeType="application/json",
            text=json.dumps(world_understanding, indent=2),
        )
    except Exception as e:
        return Resource(
            uri="memory://sleep/world-model",
            name="World Model",
            description="Error getting world model",
            mimeType="application/json",
            text=json.dumps({"error": str(e)}),
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
@rate_limited(cost=1.0)
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
                interaction_id=interaction_id,
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
        planned_actions=planned_actions,
    )

    # Log metrics
    log_tool_metric(
        "interpretation_confidence",
        confidence,
        "log_ai_interpretation",
        interaction_id=interaction_id,
        actions_count=len(planned_actions),
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
@rate_limited(cost=2.0)  # Higher cost because it triggers pattern detection
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
        interaction_log.log_feedback(feedback_type.value if feedback_type else "general", feedback)

    # Store the completed interaction (will use hybrid store if available)
    await memory.store_interaction(completed)

    # Detect patterns from this interaction
    patterns = await interaction_pattern_detector.detect_patterns([completed])

    if patterns:
        log_tool_metric(
            "patterns_detected",
            len(patterns),
            "complete_interaction",
            interaction_id=interaction_id,
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
@rate_limited(cost=1.5)  # Pattern detection is moderately expensive
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
            top_confidence=matches[0].get("confidence", 0) if matches else 0,
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
            if count
            >= len(successful_similar) * 0.5  # Action appears in 50%+ of successful interactions
        ]
        context["suggested_actions"] = suggested_actions

        # Log metrics
        log_tool_metric(
            "suggested_actions_found",
            len(suggested_actions),
            "get_relevant_context",
            successful_similar_count=len(successful_similar),
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
        # Current interaction ID is available via interaction_logger.current_interaction.interaction_id

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
            "discovery_method": (
                "interaction-based"
                if sleep_layer.role_manager.use_interaction_discovery
                else "action-based"
            ),
        }

    except Exception as e:
        return {
            "error": f"Role discovery failed: {str(e)}",
            "hint": "Ensure you have sufficient interaction history (20+ interactions)",
        }


@mcp.tool()
@mcp_tool_logger("get_world_model_proposals")
async def get_world_model_proposals(limit: int = 5) -> Dict[str, Any]:
    """
    Get goal-directed proposals based on world model understanding.

    Args:
        limit: Maximum number of proposals to return

    Returns:
        Dict containing proposals and current world understanding
    """
    if not sleep_layer:
        return {"error": "Sleep layer not enabled"}

    try:
        # Get recent interactions
        recent_interactions = await memory.store.get_recent_complete_interactions(
            hours=24, limit=50
        )

        # Get world model proposals
        proposals = await sleep_layer.get_world_model_proposals(recent_interactions)

        # Get current world understanding
        world_understanding = await sleep_layer.get_current_world_understanding()

        # Format proposals
        proposal_list = []
        for i, proposal in enumerate(proposals[:limit]):
            proposal_list.append(
                {
                    "id": f"proposal_{i}",
                    "type": proposal.type.value,
                    "description": proposal.description,
                    "rationale": proposal.rationale,
                    "confidence": proposal.confidence,
                    "suggested_actions": proposal.suggested_actions,
                    "expected_state_change": proposal.expected_state_change,
                    "alternatives": proposal.alternatives,
                }
            )

        return {
            "proposals": proposal_list,
            "world_understanding": world_understanding,
            "proposal_count": len(proposals),
            "recent_interactions_analyzed": len(recent_interactions),
        }

    except Exception as e:
        logger.error(f"Failed to get world model proposals: {e}")
        return {"error": str(e)}


@mcp.tool()
@mcp_tool_logger("provide_proposal_feedback")
async def provide_proposal_feedback(proposal_id: str, feedback: str, success: bool) -> str:
    """
    Provide feedback on a world model proposal.

    Args:
        proposal_id: ID of the proposal
        feedback: Feedback text
        success: Whether the proposal was helpful

    Returns:
        Confirmation message
    """
    if not sleep_layer:
        return "Error: Sleep layer not enabled"

    try:
        await sleep_layer.update_world_model_from_feedback(proposal_id, feedback, success)
        return f"Feedback recorded for {proposal_id}"
    except Exception as e:
        return f"Error recording feedback: {str(e)}"
