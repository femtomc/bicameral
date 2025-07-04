# LLM Module Instructions

**IMPORTANT**: This module provides LLM client implementations. All clients must follow the same interface.

## Module Purpose

The LLM module provides:
- **Unified Interface**: BaseClient abstract class
- **Multiple Providers**: OpenAI, Claude, Mock
- **Rate Limiting**: Per-provider limits
- **Error Handling**: Consistent error patterns

## Adding a New Client

**YOU MUST** implement all BaseClient methods:
```python
class YourClient(BaseClient):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Required: Implement generation
        pass

    async def count_tokens(self, text: str) -> int:
        # Required: Token counting
        pass
```

## Client Guidelines

### Error Handling
```python
try:
    response = await self._call_api(...)
except RateLimitError:
    # Re-raise for retry logic
    raise
except Exception as e:
    # Log and wrap other errors
    logger.error(f"API error: {e}")
    raise LLMError(f"Generation failed: {e}")
```

### Mock Client Usage
```python
# For testing
mock = MockClient(responses=["First", "Second"])
response = await mock.generate(request)  # Returns "First"
```

## Testing

Always test with mock client first:
```bash
pixi run python -m pytest tests/test_llm_clients.py -v
```

## Integration Points

- **LLMService**: Manages client instances
- **Sleep system**: Uses for analysis
- **Wake agent**: Uses for responses
