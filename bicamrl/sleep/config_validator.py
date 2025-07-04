"""Configuration validation for Sleep Layer."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SleepConfigValidator:
    """Validates Sleep Layer configuration."""

    REQUIRED_FIELDS = {
        "enabled": bool,
    }

    OPTIONAL_FIELDS = {
        "batch_size": (int, 10),
        "analysis_interval": (int, 300),
        "min_confidence": (float, 0.7),
        "llm_providers": (dict, {}),
        "roles": (dict, {}),
    }

    LLM_PROVIDER_FIELDS = {
        "api_key": str,
        "model": str,
        "max_tokens": (int, 4096),
        "timeout": (int, 30),
        "max_retries": (int, 3),
    }

    VALID_ROLES = ["analyzer", "generator", "enhancer", "optimizer", "synthesizer", "reviewer"]

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize Sleep Layer configuration."""
        validated = {}
        errors = []
        warnings = []

        # Check required fields
        for field, field_type in cls.REQUIRED_FIELDS.items():
            if field not in config:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(config[field], field_type):
                errors.append(f"Field {field} must be of type {field_type.__name__}")
            else:
                validated[field] = config[field]

        # Check optional fields
        for field, spec in cls.OPTIONAL_FIELDS.items():
            if isinstance(spec, tuple):
                field_type, default = spec
            else:
                field_type, default = spec, None

            if field in config:
                if not isinstance(config[field], field_type):
                    errors.append(f"Field {field} must be of type {field_type.__name__}")
                else:
                    validated[field] = config[field]
            elif default is not None:
                validated[field] = default

        # Validate LLM providers
        if "llm_providers" in validated:
            validated_providers = {}

            for provider_name, provider_config in validated["llm_providers"].items():
                if not isinstance(provider_config, dict):
                    errors.append(f"LLM provider {provider_name} must be a dictionary")
                    continue

                validated_provider = cls._validate_llm_provider(
                    provider_name, provider_config, errors, warnings
                )
                if validated_provider:
                    validated_providers[provider_name] = validated_provider

            validated["llm_providers"] = validated_providers

        # Validate role mappings
        if "roles" in validated:
            validated_roles = {}

            for role, provider in validated["roles"].items():
                if role not in cls.VALID_ROLES:
                    warnings.append(f"Unknown role: {role}")

                if provider not in validated.get("llm_providers", {}):
                    errors.append(f"Role {role} maps to unknown provider: {provider}")
                else:
                    validated_roles[role] = provider

            validated["roles"] = validated_roles

        # Validate numeric ranges
        if "batch_size" in validated and validated["batch_size"] < 1:
            errors.append("batch_size must be at least 1")

        if "analysis_interval" in validated and validated["analysis_interval"] < 10:
            errors.append("analysis_interval must be at least 10 seconds")

        if "min_confidence" in validated:
            if validated["min_confidence"] < 0 or validated["min_confidence"] > 1:
                errors.append("min_confidence must be between 0 and 1")

        # Log validation results
        if errors:
            logger.error(f"KBM config validation errors: {errors}")
            raise ValueError(f"Invalid KBM configuration: {'; '.join(errors)}")

        if warnings:
            logger.warning(f"KBM config validation warnings: {warnings}")

        return validated

    @classmethod
    def _validate_llm_provider(
        cls, name: str, config: Dict[str, Any], errors: List[str], warnings: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Validate a single LLM provider configuration."""
        validated = {}

        # Check provider-specific requirements
        if name == "claude":
            if "api_key" not in config and not config.get("use_env", True):
                errors.append("Claude provider requires api_key or use_env=true")
                return None

        elif name == "openai":
            if "api_key" not in config and not config.get("use_env", True):
                errors.append("OpenAI provider requires api_key or use_env=true")
                return None

        elif name == "local":
            if "base_url" not in config:
                config["base_url"] = "http://localhost:11434"
                warnings.append(f"Using default base_url for local provider: {config['base_url']}")

        elif name == "mock":
            # Mock provider has no requirements
            return config

        elif name == "lmstudio":
            # LM Studio uses OpenAI-compatible API
            if "base_url" not in config:
                config["base_url"] = "http://localhost:1234/v1"
                warnings.append(f"Using default base_url for LM Studio: {config['base_url']}")
            return config

        else:
            warnings.append(f"Unknown LLM provider type: {name}")

        # Validate common fields
        for field, field_type in cls.LLM_PROVIDER_FIELDS.items():
            if isinstance(field_type, tuple):
                expected_type, default = field_type
            else:
                expected_type, default = field_type, None

            if field in config:
                if not isinstance(config[field], expected_type):
                    errors.append(
                        f"Provider {name} field {field} must be of type {expected_type.__name__}"
                    )
                else:
                    validated[field] = config[field]
            elif default is not None:
                validated[field] = default

        # Copy any provider-specific fields
        for key, value in config.items():
            if key not in validated:
                validated[key] = value

        return validated

    @classmethod
    def generate_default_config(cls) -> Dict[str, Any]:
        """Generate a default KBM configuration."""
        return {
            "enabled": False,
            "batch_size": 10,
            "analysis_interval": 300,
            "min_confidence": 0.7,
            "llm_providers": {"mock": {"type": "mock"}},
            "roles": {
                "analyzer": "mock",
                "generator": "mock",
                "enhancer": "mock",
                "optimizer": "mock",
            },
        }

    @classmethod
    def save_config_template(cls, path: Path) -> None:
        """Save a configuration template to a file."""
        import json

        template = {
            "kbm": {
                "enabled": True,
                "batch_size": 10,
                "analysis_interval": 300,
                "min_confidence": 0.7,
                "llm_providers": {
                    "claude": {
                        "api_key": "${ANTHROPIC_API_KEY}",
                        "model": "claude-3-opus-20240229",
                        "max_tokens": 4096,
                    },
                    "openai": {
                        "api_key": "${OPENAI_API_KEY}",
                        "model": "gpt-4-turbo-preview",
                        "max_tokens": 4096,
                    },
                    "local": {"base_url": "http://localhost:11434", "model": "llama2"},
                },
                "roles": {
                    "analyzer": "openai",
                    "generator": "claude",
                    "enhancer": "claude",
                    "optimizer": "openai",
                },
            }
        }

        with open(path, "w") as f:
            json.dump(template, f, indent=2)

        logger.info(f"Saved KBM config template to {path}")
