"""Configuration validator for Mind.toml files."""

from pathlib import Path
from typing import Any, Dict, List

import toml


class ConfigValidator:
    """Validates Mind.toml configuration files."""

    @staticmethod
    def validate_mind_config(config: Dict[str, Any]) -> List[str]:
        """Validate Mind.toml configuration and return list of errors."""
        errors = []

        # Check for required top-level fields
        if "default_provider" not in config:
            errors.append("Missing required field: 'default_provider'")

        if "llm_providers" not in config:
            errors.append("Missing required section: '[llm_providers]'")
        else:
            providers = config["llm_providers"]
            if not isinstance(providers, dict) or not providers:
                errors.append("'llm_providers' must contain at least one provider configuration")
            else:
                # Check that default_provider exists in llm_providers
                default = config.get("default_provider")
                if default and default not in providers:
                    errors.append(
                        f"default_provider '{default}' not found in llm_providers. "
                        f"Available: {list(providers.keys())}"
                    )

                # Validate each provider
                for name, provider_config in providers.items():
                    provider_errors = ConfigValidator._validate_provider(name, provider_config)
                    errors.extend(provider_errors)

        return errors

    @staticmethod
    def _validate_provider(name: str, config: Dict[str, Any]) -> List[str]:
        """Validate a single LLM provider configuration."""
        errors = []

        # Check for provider type
        provider_type = config.get("type")
        if not provider_type:
            errors.append(f"Provider '{name}' missing required field: 'type'")
            return errors

        # Validate based on provider type
        if provider_type == "openai":
            if not config.get("api_key"):
                errors.append(f"OpenAI provider '{name}' missing required field: 'api_key'")
            if not config.get("model"):
                errors.append(f"OpenAI provider '{name}' missing required field: 'model'")

        elif provider_type == "claude":
            if not config.get("api_key"):
                errors.append(f"Claude provider '{name}' missing required field: 'api_key'")
            if not config.get("model"):
                errors.append(f"Claude provider '{name}' missing required field: 'model'")

        elif provider_type == "lmstudio":
            if not config.get("api_base"):
                errors.append(f"LMStudio provider '{name}' missing required field: 'api_base'")
            if not config.get("model"):
                errors.append(f"LMStudio provider '{name}' missing required field: 'model'")

        elif provider_type == "claude_code":
            # Claude Code doesn't require additional fields
            pass

        else:
            errors.append(f"Provider '{name}' has unknown type: '{provider_type}'")

        return errors

    @staticmethod
    def print_schema():
        """Print the expected Mind.toml schema."""
        schema = """
Mind.toml Configuration Schema
==============================

REQUIRED FIELDS:
----------------
default_provider = "provider_name"  # Must match one of the providers below

[llm_providers.provider_name]
type = "provider_type"  # One of: "openai", "claude", "lmstudio", "claude_code"
# Additional fields depend on provider type (see below)

PROVIDER TYPES:
---------------

1. OpenAI Provider:
   [llm_providers.my_openai]
   type = "openai"
   api_key = "${OPENAI_API_KEY}"  # Or direct key
   model = "gpt-4o-mini"          # Required
   temperature = 0.7              # Optional (default: 0.7)
   max_tokens = 2048              # Optional

2. Claude Provider:
   [llm_providers.my_claude]
   type = "claude"
   api_key = "${ANTHROPIC_API_KEY}"  # Or direct key
   model = "claude-3-5-sonnet-latest"  # Required
   temperature = 0.7                    # Optional
   max_tokens = 2048                    # Optional

3. LM Studio Provider:
   [llm_providers.my_lmstudio]
   type = "lmstudio"
   api_base = "http://localhost:1234/v1"  # Required
   model = "local-model-name"             # Required
   temperature = 0.7                      # Optional

4. Claude Code Provider:
   [llm_providers.my_claude_code]
   type = "claude_code"
   enabled = true         # Optional
   temperature = 0.7      # Optional
   max_tokens = 2048      # Optional

EXAMPLE CONFIGURATION:
----------------------
# Use Claude Code as default
default_provider = "claude_code"

[llm_providers.claude_code]
type = "claude_code"
enabled = true

# Alternative providers
[llm_providers.openai]
type = "openai"
api_key = "${OPENAI_API_KEY}"
model = "gpt-4o-mini"

[llm_providers.local]
type = "lmstudio"
api_base = "http://localhost:1234/v1"
model = "mistral-7b"
"""
        print(schema)

    @staticmethod
    def validate_file(file_path: Path) -> bool:
        """Validate a Mind.toml file and print errors."""
        try:
            config = toml.load(file_path)
        except Exception as e:
            print(f"Error parsing TOML file: {e}")
            return False

        errors = ConfigValidator.validate_mind_config(config)

        if errors:
            print("Configuration errors found:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("Configuration is valid!")
            return True
