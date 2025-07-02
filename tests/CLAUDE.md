# Bicamrl Testing Guide

This guide contains instructions for testing Bicamrl with Claude Desktop.

## Production Testing Checklist

### Setup
- [ ] Copy `.bicamrl/Mind.toml` to your home directory: `~/.bicamrl/Mind.toml`
- [ ] Set `OPENAI_API_KEY` environment variable (or use ${OPENAI_API_KEY} in Mind.toml)
- [ ] Configure Claude Desktop to use Bicamrl
- [ ] Start monitoring script: `python scripts/monitor_production_test.py`

### Test Scenarios

#### 1. Debugging Session (Day 1)
- [ ] Find and fix a real bug in a project
- [ ] Let Bicamrl track the debugging workflow
- [ ] Note any errors or unexpected behavior
- [ ] Check if a "Debug Specialist" role emerges

#### 2. Feature Implementation (Day 1-2)
- [ ] Implement a small feature from scratch
- [ ] Use multiple files and test-driven development
- [ ] Observe how context suggestions work
- [ ] See if "Feature Builder" or similar role emerges

#### 3. Code Refactoring (Day 2)
- [ ] Refactor some existing code
- [ ] Make multiple small changes
- [ ] Test pattern detection for refactoring workflows
- [ ] Check if refactoring patterns are learned

#### 4. Documentation Writing (Day 2-3)
- [ ] Write or update documentation
- [ ] See how Bicamrl handles non-code tasks
- [ ] Check if documentation patterns emerge

#### 5. Code Review (Day 3)
- [ ] Review some code (yours or others)
- [ ] Make review comments and suggestions
- [ ] Test if review patterns are captured

### Monitoring Points

#### After Each Session
- [ ] Run `discover_roles` tool to check new roles
- [ ] Check `@bicamrl/sleep/insights` for insights
- [ ] Review `@bicamrl/patterns` for new patterns
- [ ] Check logs for errors: `tail -f logs/bicamrl.log` (or path in Mind.toml)

#### Daily Checks
- [ ] Memory stats: `get_memory_stats` tool
- [ ] Role statistics: `@bicamrl/sleep/roles/statistics`
- [ ] Performance: Note any slow operations
- [ ] Errors: Document any crashes or exceptions

### Data Collection

#### Performance Metrics
- Response time for tools (should be <100ms)
- Memory growth over time
- Pattern detection accuracy
- Role activation correctness

#### Quality Metrics
- Are suggested contexts relevant?
- Do discovered roles make sense?
- Are patterns actually useful?
- Does prompt optimization help?

#### Issues to Track
- [ ] Errors and stack traces
- [ ] Slow operations
- [ ] Incorrect pattern matches
- [ ] Missing functionality
- [ ] Confusing behavior

### End of Test

#### Analysis (Day 3-4)
- [ ] Export discovered roles
- [ ] Document most useful patterns
- [ ] List performance bottlenecks
- [ ] Prioritize bugs to fix
- [ ] Identify missing features

#### Report
Create a summary with:
1. Discovered roles and their effectiveness
2. Most valuable patterns learned
3. Performance issues found
4. Bugs encountered
5. Feature requests
6. Overall assessment

### Tips

1. **Use Bicamrl naturally** - Don't try to "test" it, just use it
2. **Be patient** - Patterns need 3+ occurrences to be detected
3. **Give feedback** - Use `record_feedback` when something is wrong
4. **Check monitoring** - Run the monitor script in another terminal
5. **Take notes** - Document anything surprising or broken

## Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bicamrl

# Run specific test files
pytest tests/test_mcp_compliance.py
pytest tests/test_mcp_integration.py

# Run with verbose output
pytest -v
```

## Automated Production Testing

You can run an automated production test that simulates real usage:

```bash
# Run full production test
pixi run test-production

# Run quick test (subset of scenarios)
pixi run test-production-quick

# Monitor during test (in another terminal)
pixi run test-monitor

# Use custom config (directly with Python)
python scripts/run_production_test.py --config /path/to/Mind.toml
```

### Using LM Studio (Local Models)

You can run production tests with local models using LM Studio:

1. Start LM Studio and load a model
2. Start the local server (usually port 1234)
3. Note the model name shown in LM Studio
4. Run the test with the model name:
   ```bash
   # Quick test with LM Studio
   pixi run test-lmstudio "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

   # Full test with LM Studio
   pixi run test-lmstudio-full "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
   ```

See `docs/LM_STUDIO_SETUP.md` for detailed instructions.

The production test simulates:
- Debugging workflows
- Feature implementation
- Code refactoring
- Code reviews
- Documentation writing
- Pattern learning
- Role discovery

Results are saved to `production_test_results.json` with metrics on:
- Response times
- Pattern detection accuracy
- Role discovery effectiveness
- Memory growth
- Error tracking

## Test Structure

- `test_mcp_compliance.py` - Tests MCP protocol compliance
- `test_mcp_integration.py` - Tests end-to-end MCP integration
- `test_memory_manager.py` - Tests memory management functionality
- `test_pattern_detector.py` - Tests pattern detection
- `test_feedback_processor.py` - Tests feedback processing
- `test_interaction_logger.py` - Tests interaction logging
- `test_hybrid_store.py` - Tests hybrid storage (SQLite + vector)
- `test_sleep_layer.py` - Tests Sleep meta-cognitive layer
- `test_prompt_optimizer.py` - Tests prompt optimization
- `test_role_manager.py` - Tests role management
- `test_interaction_role_discoverer.py` - Tests role discovery from interactions
- `test_stress_performance.py` - Performance and stress tests

## Key Testing Areas

### Memory System
- Pattern detection accuracy
- Memory consolidation
- Context retrieval relevance
- Search performance

### Sleep Layer
- Insight generation quality
- Role discovery effectiveness
- Prompt optimization improvements
- LLM provider integration

### Interaction Model
- Complete interaction tracking
- Action logging accuracy
- Feedback processing
- Pattern detection from interactions

### Performance
- Tool response times (<100ms target)
- Memory growth over time
- Concurrent operation handling
- Large dataset performance
