# Testing with LM Studio

## Setup

1. **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/)

2. **Download a Model**:
   - Open LM Studio
   - Go to the "Search" tab
   - Download a model like:
     - `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` (recommended for testing)
     - `TheBloke/CodeLlama-7B-Instruct-GGUF` (good for code tasks)
     - Any other GGUF model that fits your system

3. **Start the Server**:
   - Go to "Local Server" tab in LM Studio
   - Select your downloaded model
   - Click "Start Server"
   - Default URL is `http://localhost:1234/v1`

## Running Tests

### Quick Test of LM Studio Connection
```bash
pixi run python -m pytest tests/test_lmstudio_connection.py -v -s
```

### Run All Tests with Mock LLM (Fast)
```bash
pixi run test
```

### Run Specific Tests with LM Studio
```bash
# Test world model with real LLM
pixi run python -m pytest tests/test_world_model.py -v -s -k "real_llm"

# Run all LM Studio tests
pixi run python -m pytest -v -s -m lmstudio
```

### Run Tests with Different Providers
```bash
# Force mock LLM (no external dependencies)
pixi run python -m pytest -v -m mock

# Use OpenAI (requires OPENAI_API_KEY)
OPENAI_API_KEY=your-key pixi run python -m pytest -v -m openai

# Use specific LM Studio model
LMSTUDIO_MODEL="mistral-7b-instruct" pixi run python -m pytest -v -m lmstudio
```

## Environment Variables

- `LMSTUDIO_BASE_URL`: LM Studio server URL (default: `http://localhost:1234/v1`)
- `LMSTUDIO_MODEL`: Model name to use (default: `local-model`)
- `OPENAI_API_KEY`: OpenAI API key (only needed for OpenAI tests)

## Test Markers

- `@pytest.mark.mock` - Use mock LLM (fastest, no dependencies, default)
- `@pytest.mark.lmstudio` - Use LM Studio (requires running server)
- `@pytest.mark.openai` - Use OpenAI API (requires API key)
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.integration` - Integration tests

**Note**: By default, tests use the mock LLM provider to ensure they can run without external dependencies. Only tests explicitly marked with `@pytest.mark.lmstudio` or `@pytest.mark.openai` will try to use real LLMs.

## Writing Tests

```python
import pytest

@pytest.mark.asyncio
@pytest.mark.mock  # Fast test with mock
async def test_something_quick(memory):
    # This will use mock LLM responses
    result = await memory.pattern_detector.check_for_patterns()
    assert result is not None

@pytest.mark.asyncio
@pytest.mark.lmstudio  # Test with real local LLM
async def test_something_with_quality(memory):
    # This will use LM Studio for real LLM inference
    # Test will be skipped if LM Studio is not running
    world_model = await memory.consolidator.consolidate_memories()
    # Check quality of results...
```

## Troubleshooting

### LM Studio Tests Skipped
If tests marked with `@pytest.mark.lmstudio` are skipped:
1. Check LM Studio is running
2. Check a model is loaded
3. Check server is started on correct port
4. Try the connection test first

### Slow Tests
Local models can be slow. To speed up:
1. Use smaller models (7B parameters)
2. Reduce `max_tokens` in config
3. Use `@pytest.mark.mock` for most tests
4. Only use `@pytest.mark.lmstudio` for quality checks

### Out of Memory
If you get OOM errors:
1. Use smaller models
2. Reduce context length in LM Studio settings
3. Close other applications

### JSON Parsing Errors
Many local models don't reliably produce JSON output. The system has fallback parsing that:
1. Tries to parse as JSON first
2. Falls back to extracting information from natural language
3. Uses heuristics to identify domains and entities

For best results with local models, consider using models fine-tuned for instruction following or JSON output.
