# Storage Module Instructions

**IMPORTANT**: This module handles all data persistence. Data integrity is paramount.

## Module Purpose

The storage module provides:
- **SQLite Persistence**: Primary storage with automatic migrations
- **Archive System**: Historical data with interaction tracking
- **Vector Store**: Optional embeddings for semantic search
- **Hybrid Store**: Combines SQLite + vectors when available
- **JSON Safety**: Protects against JSON parsing edge cases

## Critical Architecture Decisions

### Database Schema
**YOU MUST**:
- NEVER manually modify schema - use migrations
- ALWAYS archive data before deletion
- Track archive_reason for audit trails
- Use interaction_id for correlation

### Archive Tables
```sql
-- Main tables
interactions -> archived_interactions
patterns -> archived_patterns
world_model_states -> (snapshot system)

-- Archive reasons
'consolidated_to_working'
'consolidated_to_episodic'
'promoted_to_semantic'
```

### Transaction Boundaries
**CRITICAL**: Use transactions for consistency:
```python
async with store.transaction():
    await store.add_interaction(...)
    await store.update_pattern(...)
    # Commits on success, rolls back on error
```

## Key Files

- `sqlite_store.py` - Main database interface
- `hybrid_store.py` - SQLite + vector combination
- `vector_store.py` - Embedding storage (optional)
- `llm_embeddings.py` - LLM-based embeddings
- `json_utils.py` - Safe JSON handling

## Common Operations

### Basic storage
```python
store = SQLiteStore(db_path)
await store.initialize()

# Add data
await store.add_interaction(interaction_data)
await store.add_pattern(pattern_data)

# Query data
interactions = await store.get_recent_interactions(limit=10)
patterns = await store.get_patterns_by_type("workflow")
```

### Archive operations
```python
# Archive old interactions
await store.archive_old_interactions(
    before_date=cutoff,
    reason="consolidated_to_working"
)

# Query archives
archived = await store.get_archived_interactions(
    interaction_id="uuid-here"
)
```

### World model persistence
```python
# Save world state
await store.add_world_model_state({
    "domain": "coding",
    "entities": [...],
    "relations": [...],
    "confidence": 0.8
})

# Get snapshots
snapshots = await store.get_world_model_snapshots(limit=5)
```

## Schema Management

**IMPORTANT**: Schema changes are automatic:
1. Add new column to CREATE TABLE statement
2. Add migration in `_ensure_schema()`
3. Test with existing databases

Example migration:
```python
# Check if column exists
cursor.execute("PRAGMA table_info(interactions)")
columns = [col[1] for col in cursor.fetchall()]
if 'new_column' not in columns:
    cursor.execute("ALTER TABLE interactions ADD COLUMN new_column TEXT")
```

## Testing

Run storage tests:
```bash
pixi run python -m pytest tests/test_storage.py -v
pixi run python -m pytest tests/test_sqlite_store_json_safety.py -v
pixi run python -m pytest tests/test_hybrid_store.py -v
```

## Common Pitfalls

- **Missing await**: All operations are async
- **No transactions**: Use transaction context manager
- **Raw SQL**: Use parameterized queries
- **Schema conflicts**: Let migrations handle changes
- **JSON edge cases**: Use json_utils for safety

## Performance Tips

- Use indexes on frequently queried columns
- Batch operations when possible
- Archive old data regularly
- VACUUM occasionally for space recovery
- Monitor database size growth

## Integration Points

- **Core memory**: Primary storage backend
- **Pattern detector**: Stores discovered patterns
- **World model**: Persists world knowledge
- **Sleep system**: Reads for analysis
