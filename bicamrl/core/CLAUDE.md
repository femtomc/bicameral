# Core Module Instructions

**IMPORTANT**: This module contains the heart of Bicamrl's memory and pattern detection systems. Read this before making changes.

## Module Purpose

The core module implements:
- **Hierarchical Memory System**: Active → Working → Episodic → Semantic
- **Pattern Detection**: Dynamic discovery of user workflows
- **World Model**: LLM-driven entity and relationship discovery
- **Feedback Processing**: Developer corrections and preferences

## Critical Architecture Decisions

### Memory Hierarchy
**IMPORTANT**: These thresholds are COUNT-BASED, not time-based:
```python
active_to_working_threshold = 10    # 10 interactions → working memory
working_to_episodic_threshold = 5   # 5 working memories → episodic
episodic_to_semantic_threshold = 10 # 10 episodes → semantic
min_frequency_for_semantic = 5      # Pattern frequency for semantic
```

### World Model
**YOU MUST**:
- NEVER hardcode domains, entities, or patterns
- Let the LLM discover all world knowledge dynamically
- Store discovered knowledge in dedicated world_model tables
- Use JSON prompts optimized for instruction-tuned models

### Database Operations
**CRITICAL**: All database operations must:
- Use proper transaction boundaries
- Archive data before deletion (with interaction_id tracking)
- Handle schema migrations automatically in SQLiteStore
- Maintain referential integrity across archives

## Key Files

- `memory.py` - Main memory orchestrator
- `memory_consolidator.py` - Handles memory promotion logic
- `pattern_detector.py` - Discovers action patterns
- `interaction_pattern_detector.py` - NLP-based pattern discovery
- `world_model.py` - Dynamic world modeling
- `llm_service.py` - Centralized LLM operations
- `feedback_processor.py` - Processes user corrections

## Common Operations

### Adding a new memory
```python
memory = Memory(db_path)
await memory.add_interaction(user_query, ai_interpretation, actions)
```

### Forcing consolidation
```python
await memory.consolidate_memories()  # Runs all consolidation steps
```

### Querying world model
```python
world_state = await memory.world_model.get_world_state()
relevant_context = await memory.world_model.get_relevant_context(query)
```

## Testing

Run module-specific tests:
```bash
pixi run python -m pytest tests/test_memory.py -v
pixi run python -m pytest tests/test_pattern_detector.py -v
pixi run python -m pytest tests/test_world_model.py -v
```

## Common Pitfalls

- **Forgetting async**: All I/O operations must be async
- **Hardcoding patterns**: Let the LLM discover patterns
- **Time-based logic**: Use interaction counts, not timestamps
- **Missing archives**: Always archive before deletion
- **Skipping consolidation**: Test with forced consolidation

## Integration Points

- **Storage layer**: Uses SQLiteStore for persistence
- **LLM service**: All AI operations go through LLMService
- **Sleep system**: Provides patterns for sleep analysis
- **MCP server**: Exposes memory operations as tools
