"""Configuration loader for bicamrl."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .utils.logging_config import get_logger

logger = get_logger("config_loader")


def load_config() -> Dict[str, Any]:
    """Load configuration from various sources in priority order."""
    config = {}

    # Default configuration
    default_config = {
        "memory_db_path": str(Path.home() / ".bicamrl" / "memory"),
        "vector_store": {"backend": "basic"},
        "pattern_detection": {
            "min_frequency": 3,
            "confidence_threshold": 0.6,
            "recency_weight_days": 7,
            "sequence_max_length": 5,
        },
        "memory": {
            "active_to_working_threshold": 10,
            "working_to_episodic_threshold": 5,
            "episodic_to_semantic_threshold": 10,
        },
        "logging": {"level": "INFO"},
    }

    # Start with defaults
    config.update(default_config)

    # Load from config files (in order of priority)
    config_paths = [
        Path.home() / ".bicamrl" / "config.json",  # User global
        Path("bicamrl_config.json"),  # Project root
        Path(".bicamrl") / "config.json",  # Project-specific
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                    # Deep merge
                    config = deep_merge(config, file_config)
                    logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")

    # Override with environment variables
    env_overrides = {
        "MEMORY_DB_PATH": ["memory_db_path"],
        "VECTOR_BACKEND": ["vector_store", "backend"],
        "PATTERN_MIN_FREQUENCY": ["pattern_detection", "min_frequency"],
        "PATTERN_CONFIDENCE_THRESHOLD": ["pattern_detection", "confidence_threshold"],
        "BICAMRL_LOG_LEVEL": ["logging", "level"],
    }

    for env_var, config_path in env_overrides.items():
        value = os.environ.get(env_var)
        if value:
            set_nested(config, config_path, value)
            logger.debug(f"Overrode {'.'.join(config_path)} with ${env_var}")

    return config


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def set_nested(data: Dict[str, Any], path: list, value: Any):
    """Set a value in a nested dictionary using a path."""
    current = data
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Convert string values to appropriate types
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        elif "." in value and all(part.isdigit() for part in value.split(".", 1)):
            value = float(value)

    current[path[-1]] = value


def get_vector_backend(config: Optional[Dict[str, Any]] = None) -> str:
    """Get the configured vector backend."""
    if config is None:
        config = load_config()

    return config.get("vector_store", {}).get("backend", "basic")


def get_memory_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Get the configured memory database path."""
    if config is None:
        config = load_config()

    path = config.get("memory_db_path", str(Path.home() / ".bicamrl" / "memory"))
    return Path(path).expanduser()
