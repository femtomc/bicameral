# Configuration Guide

This guide covers all configuration options for Bicamrl.

## Configuration Files

Bicamrl looks for configuration in these locations (in order):
1. `.bicamrl/config.json` (project-specific)
2. `bicamrl_config.json` (project root)
3. `~/.bicamrl/config.json` (user global)
4. Environment variables

## Basic Configuration

### Minimal Setup
No configuration required! Bicamrl works with sensible defaults.

### Custom Memory Location
```json
{
  "memory_db_path": "/custom/path/to/memory"
}
```

Or via environment variable:
```bash
export MEMORY_DB_PATH=/custom/path/to/memory
```

## Pattern Detection Settings

```json
{
  "pattern_detection": {
    "min_frequency": 3,              // Minimum occurrences to detect pattern
    "confidence_threshold": 0.6,     // Minimum confidence to store pattern
    "recency_weight_days": 7,        // Weight recent patterns higher
    "sequence_max_length": 5         // Maximum pattern sequence length
  }
}
```

## Memory Management

```json
{
  "memory": {
    "active_to_working_hours": 1,    // Hours before consolidation
    "working_to_episodic_days": 1,   // Days before episodic
    "episodic_to_semantic_days": 7,  // Days before semantic
    "max_interactions": 100000,      // Maximum stored interactions
    "consolidation_interval": 3600   // Seconds between consolidations
  }
}
```

## Sleep Layer (Background Processing)

### Basic Sleep Layer Setup

```json
{
  "kbm": {
    "enabled": true,
    "llm_providers": {
      "openai": {
        "api_key": "${OPENAI_API_KEY}",  // Or use actual key
        "model": "gpt-4-turbo-preview",
        "max_tokens": 4096
      }
    }
  }
}
```

### Advanced Sleep Layer Configuration

```json
{
  "kbm": {
    "enabled": true,
    "batch_size": 10,                // Observations per batch
    "analysis_interval": 300,        // Seconds between analyses
    "min_confidence": 0.7,           // Minimum insight confidence

    "llm_providers": {
      "openai": {
        "api_key": "${OPENAI_API_KEY}",
        "model": "gpt-4-turbo-preview",
        "max_tokens": 4096,
        "timeout": 30,
        "max_retries": 3
      },
      "claude": {
        "api_key": "${ANTHROPIC_API_KEY}",
        "model": "claude-3-opus-20240229",
        "max_tokens": 4096
      },
      "local": {
        "base_url": "http://localhost:11434",
        "model": "llama2"
      }
    },

    "roles": {
      "analyzer": "openai",      // Pattern analysis
      "generator": "claude",     // Content generation
      "enhancer": "claude",      // Prompt enhancement
      "optimizer": "openai"      // Optimization
    }
  }
}
```

## Environment Variables

All configuration options can be set via environment:

```bash
# Basic settings
export MEMORY_DB_PATH=~/.bicamrl/memory
export BICAMERAL_LOG_LEVEL=INFO

# Sleep Layer settings
export SLEEP_LAYER_ENABLED=true
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Pattern detection
export PATTERN_MIN_FREQUENCY=3
export PATTERN_CONFIDENCE_THRESHOLD=0.6
```

## Logging Configuration

```json
{
  "logging": {
    "level": "INFO",              // DEBUG, INFO, WARNING, ERROR
    "log_dir": ".bicamrl/logs",
    "max_file_size": "10MB",
    "max_files": 5,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

## Performance Tuning

### For Large Codebases
```json
{
  "performance": {
    "cache_size": 1000,
    "db_connection_pool": 5,
    "pattern_check_interval": 600,  // Check less frequently
    "max_context_files": 20         // Limit context size
  }
}
```

### For Small Projects
```json
{
  "performance": {
    "pattern_check_interval": 60,   // Check more frequently
    "min_frequency": 2,             // Lower pattern threshold
    "consolidation_interval": 1800  // Faster consolidation
  }
}
```

## Security Settings

```json
{
  "security": {
    "encrypt_memory": false,        // Enable database encryption
    "audit_logging": true,          // Log all access
    "sanitize_paths": true,         // Remove sensitive path info
    "excluded_patterns": [          // Never log these
      "*.key",
      "*.pem",
      "**/secrets/**"
    ]
  }
}
```

## Complete Example

Here's a complete configuration for a development team:

```json
{
  "memory_db_path": "~/.bicamrl/team_memory",

  "pattern_detection": {
    "min_frequency": 3,
    "confidence_threshold": 0.7,
    "recency_weight_days": 14
  },

  "memory": {
    "consolidation_interval": 7200,
    "max_interactions": 50000
  },

  "kbm": {
    "enabled": true,
    "analysis_interval": 600,
    "llm_providers": {
      "openai": {
        "api_key": "${OPENAI_API_KEY}",
        "model": "gpt-4-turbo-preview"
      }
    },
    "roles": {
      "analyzer": "openai",
      "generator": "openai"
    }
  },

  "logging": {
    "level": "INFO",
    "log_dir": "/var/log/bicamrl"
  },

  "security": {
    "audit_logging": true,
    "excluded_patterns": ["**/.env", "**/*.key"]
  }
}
```

## Validation

To validate your configuration:

```bash
python -m bicamrl.validate_config bicamrl_config.json
```

This will check for:
- Valid JSON syntax
- Required fields
- Type correctness
- API key format (without revealing keys)
- File path accessibility
