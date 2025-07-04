# Utils Module Instructions

**IMPORTANT**: This module provides shared utilities. Keep dependencies minimal.

## Module Purpose

The utils module provides:
- **Structured Logging**: JSON logs for production debugging
- **MCP Decorators**: Automatic logging for MCP tools
- **Rate Limiting**: Token bucket implementation
- **Embeddings**: Text processing utilities
- **Configuration**: Validation and loading

## Critical Patterns

### Logging Infrastructure
**YOU MUST** use structured logging:
```python
from .logging_config import get_logger

logger = get_logger(__name__)
logger.info("Operation completed", extra={
    "interaction_id": "uuid",
    "duration_ms": 150,
    "pattern_count": 3
})
```

### MCP Tool Decorators
**ALWAYS** use decorators for MCP tools:
```python
from .mcp_logging import mcp_tool_logger

@mcp.tool()
@mcp_tool_logger("my_tool")
async def my_tool(param: str) -> dict:
    # Automatically logs entry, exit, errors, timing
    return {"status": "success"}
```

### Rate Limiting
```python
from .rate_limit_decorator import with_rate_limit

@with_rate_limit(calls=10, period=60, cost=1)
async def expensive_operation():
    pass
```

## Key Files

- `logging_config.py` - JSON structured logging setup
- `mcp_logging.py` - MCP-specific decorators
- `log_utils.py` - Logging context managers
- `rate_limit_decorator.py` - Token bucket rate limiting
- `embeddings.py` - Text similarity utilities
- `config_validator.py` - Mind.toml validation

## Logging Best Practices

### Use Context Managers
```python
from .log_utils import log_operation

async with log_operation("pattern_detection", logger):
    # Automatically logs start, end, duration, errors
    patterns = await detect_patterns()
```

### Add Structured Context
```python
logger.info("Pattern detected", extra={
    "pattern_type": "workflow",
    "confidence": 0.85,
    "action_count": 5
})
```

### Log Levels
- DEBUG: Detailed debugging info
- INFO: Normal operations
- WARNING: Recoverable issues
- ERROR: Errors that need attention

## Testing

Run utils tests:
```bash
pixi run python -m pytest tests/test_rate_limiter.py -v
pixi run python -m pytest tests/test_config_validator.py -v
```

## Common Pitfalls

- **Import cycles**: Utils should not import from other modules
- **Heavy dependencies**: Keep utils lightweight
- **Global state**: Avoid module-level state
- **Sync functions**: Prefer async for consistency

## Integration Points

- **All modules**: Logging infrastructure
- **MCP server**: Tool decorators
- **LLM service**: Rate limiting
- **Config loader**: Validation
