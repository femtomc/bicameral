# Sleep Module Instructions

**IMPORTANT**: The sleep system is Bicamrl's meta-cognitive layer. It runs in the background to continuously improve the system.

## Module Purpose

The sleep module implements:
- **Role Discovery**: LLM-driven behavioral role identification
- **Multi-LLM Coordination**: Different models for different tasks
- **Prompt Optimization**: Learning better prompts from patterns
- **Continuous Improvement**: Background analysis of interactions

## Critical Architecture Decisions

### Background Processing
**YOU MUST**:
- Run sleep analysis asynchronously (never block wake operations)
- Respect the analysis_interval (default: 120 seconds)
- Use batch_size for processing (default: 5 interactions)
- Maintain min_confidence thresholds (default: 0.6)

### LLM Provider Management
**IMPORTANT**: Sleep can use different LLMs than wake:
```toml
[sleep.llm_providers.openai]
type = "openai"
model = "gpt-4"

[sleep.llm_providers.lmstudio]
type = "lmstudio"
api_base = "http://localhost:1234/v1"
```

### Role System
**CRITICAL**: Roles are discovered, not hardcoded:
- Stored as Markdown in `~/.bicamrl/roles/`
- Include both `.md` (human-readable) and `.json` (structured)
- Discovered through LLM analysis of patterns
- Can be manually edited by users

## Key Files

- `sleep.py` - Main sleep orchestrator
- `role_manager.py` - Manages discovered roles
- `role_proposer.py` - Proposes new roles from patterns
- `llm_providers.py` - Multi-LLM provider support
- `prompt_optimizer.py` - Learns better prompts
- `config_validator.py` - Validates Mind.toml

## Common Operations

### Starting sleep system
```python
sleep = Sleep(memory, config)
await sleep.start()  # Runs in background
```

### Manually triggering analysis
```python
await sleep.analyze_patterns()
await sleep.discover_roles()
```

### Checking discovered roles
```python
roles = sleep.role_manager.list_roles()
role = sleep.role_manager.get_role("code_reviewer")
```

## Configuration

In Mind.toml:
```toml
[sleep]
enabled = true
batch_size = 5
analysis_interval = 120
min_confidence = 0.6
discovery_interval = 300

[sleep.llm_providers.your_provider]
type = "openai"  # or "lmstudio", "anthropic"
# Provider-specific settings
```

## Testing

Run sleep-specific tests:
```bash
pixi run python -m pytest tests/test_sleep_layer_behavior.py -v
pixi run python -m pytest tests/test_role_manager.py -v
pixi run python -m pytest tests/test_role_proposer.py -v
```

Test with LM Studio:
```bash
pixi run test-lmstudio "your-model-name"
```

## Common Pitfalls

- **Blocking operations**: Sleep must never block wake
- **Hardcoded roles**: All roles must be discovered
- **Missing providers**: Check LLM providers are configured
- **Rate limits**: Respect provider rate limits
- **Resource usage**: Monitor background CPU/memory

## Integration Points

- **Memory core**: Reads patterns and interactions
- **Storage layer**: Persists discovered roles
- **Wake layer**: Provides optimized prompts
- **LLM service**: Coordinates multiple providers
