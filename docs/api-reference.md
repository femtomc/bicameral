# API Reference

Complete reference for Bicamrl's MCP tools and resources.

## MCP Tools

### log_interaction

Records an interaction with the codebase.

**Parameters:**
- `action` (string, required): The action performed
- `file_path` (string, optional): Path to the file
- `details` (object, optional): Additional metadata

**Returns:** Confirmation message with pattern detection count

**Example:**
```python
await log_interaction(
    action="edit_file",
    file_path="src/main.py",
    details={"lines_changed": 42}
)
```

### detect_pattern

Checks if an action sequence matches known patterns.

**Parameters:**
- `action_sequence` (array, required): List of actions to match

**Returns:** Array of matching patterns with confidence scores

**Example:**
```python
matches = await detect_pattern(
    action_sequence=["open_file", "edit_file", "save_file"]
)
# Returns: [{"pattern": {...}, "match_type": "exact", "confidence": 0.9}]
```

### get_relevant_context

Retrieves context relevant to the current task.

**Parameters:**
- `task_description` (string, required): Description of the task
- `file_context` (array, optional): Currently relevant files

**Returns:** Object with patterns, preferences, and suggestions

**Example:**
```python
context = await get_relevant_context(
    task_description="implement user authentication",
    file_context=["auth.py", "models/user.py"]
)
```

### record_feedback

Records developer feedback to improve behavior.

**Parameters:**
- `feedback_type` (string, required): One of "correct", "prefer", "pattern"
- `message` (string, required): Feedback content

**Returns:** Confirmation of feedback recording

**Example:**
```python
await record_feedback(
    feedback_type="prefer",
    message="always use type hints in function signatures"
)
```

### search_memory

Searches through stored memories.

**Parameters:**
- `query` (string, required): Search query
- `limit` (integer, optional): Maximum results (default: 10)

**Returns:** Array of search results

**Example:**
```python
results = await search_memory(
    query="authentication",
    limit=5
)
```

### get_memory_stats

Retrieves memory system statistics.

**Parameters:** None

**Returns:** Statistics object

**Example:**
```python
stats = await get_memory_stats()
# Returns: {
#   "total_interactions": 1523,
#   "total_patterns": 47,
#   "active_sessions": 3,
#   "top_files": ["main.py", "tests.py"]
# }
```

### consolidate_memories

Runs memory consolidation process.

**Parameters:** None

**Returns:** Consolidation statistics

**Example:**
```python
result = await consolidate_memories()
# Returns: {
#   "active_to_working": 15,
#   "working_to_episodic": 8,
#   "episodic_to_semantic": 3
# }
```

### get_memory_insights

Gets insights relevant to a context.

**Parameters:**
- `context` (string, required): Context to analyze

**Returns:** Insights object with memories and patterns

### optimize_prompt (Sleep required)

Enhances a prompt based on past interactions.

**Parameters:**
- `prompt` (string, required): Original prompt
- `task_type` (string, optional): Type of task
- `context` (object, optional): Current context

**Returns:** Enhanced prompt with reasoning

### observe_interaction (Sleep required)

Reports an interaction to the Sleep for analysis.

**Parameters:**
- `interaction_type` (string, required): Type of interaction
- `query` (string, required): The query/prompt
- `response` (string, required): The response
- `tokens_used` (integer, optional): Token count
- `latency` (float, optional): Response time
- `success` (boolean, optional): Success status

### get_sleep_layer_recommendation (Sleep required)

Gets recommendations from the Sleep.

**Parameters:**
- `query` (string, required): What to get recommendations for
- `context` (object, optional): Current context

## MCP Resources

### Health Check Resources

#### @bicamrl/health/status

Comprehensive health status of all server components.

**Format:** JSON object

**Example:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600.5,
  "check_duration_ms": 15.2,
  "components": {
    "memory": {
      "status": "healthy",
      "latency_ms": 2.3,
      "total_interactions": 1234,
      "total_patterns": 45
    },
    "storage": {
      "status": "healthy",
      "sqlite": {
        "status": "healthy",
        "exists": true,
        "size_mb": 12.5
      },
      "vector_store": {
        "status": "healthy",
        "backend": "LanceDBVectorStore",
        "has_embeddings": true
      }
    },
    "sleep_layer": {
      "status": "healthy",
      "is_running": true,
      "queue_size": 3,
      "insights_cached": 12
    }
  },
  "version": "1.0.0"
}
```

#### @bicamrl/health/ready

Readiness check - indicates if server can handle requests.

**Format:** JSON object

**Example:**
```json
{
  "ready": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "memory": true,
    "pattern_detector": true,
    "feedback_processor": true,
    "interaction_logger": true
  }
}
```

#### @bicamrl/health/live

Simple liveness check - indicates server is running.

**Format:** JSON object

**Example:**
```json
{
  "alive": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "server": "bicamrl",
  "version": "1.0.0"
}
```

#### @bicamrl/health/rate-limits

Current rate limiting metrics and configuration.

**Format:** JSON object

**Example:**
```json
{
  "total_requests": 1234,
  "rejected_requests": 12,
  "rejection_rate": 0.0097,
  "active_clients": 5,
  "config": {
    "requests_per_minute": 60,
    "burst_size": 10,
    "window_seconds": 60
  }
}
```

### Pattern Resources

#### @bicamrl/patterns

All learned patterns from interactions.

**Format:** JSON array of pattern objects

**Pattern Object:**
```json
{
  "name": "Edit-Test-Commit workflow",
  "description": "Common development cycle",
  "pattern_type": "workflow",
  "sequence": ["edit", "test", "commit"],
  "frequency": 23,
  "confidence": 0.87,
  "last_seen": "2024-01-15T10:30:00Z"
}
```

### @bicamrl/preferences

Developer preferences organized by category.

**Format:** JSON object with categories

**Example:**
```json
{
  "style": {
    "indentation": "4 spaces",
    "quotes": "double"
  },
  "testing": {
    "framework": "pytest",
    "coverage_threshold": 80
  }
}
```

### @bicamrl/context/recent

Recent activity context.

**Format:** JSON object

**Example:**
```json
{
  "session_id": "2024-01-15T09:00:00Z",
  "recent_actions": [...],
  "top_files": [
    {"file": "main.py", "count": 15},
    {"file": "tests.py", "count": 8}
  ],
  "total_interactions": 47
}
```

### @bicamrl/sleep/insights (Sleep only)

Insights from the Sleep.

**Format:** JSON array of insight objects

### @bicamrl/sleep/status (Sleep only)

Sleep operational status.

**Format:** JSON object

**Example:**
```json
{
  "enabled": true,
  "is_running": true,
  "config": {...},
  "statistics": {
    "insights_cached": 12,
    "observation_queue_size": 3
  }
}
```

### @bicamrl/sleep/config (Sleep only)

Current Sleep configuration.

### @bicamrl/sleep/prompt-templates (Sleep only)

Optimized prompt templates.

## Pattern Types

### workflow
Multi-step sequences of actions, like "edit-test-commit"

### file_access
File access patterns, like "always open config.json before main.py"

### action_sequence
Repeated action sequences regardless of files

### error
Common error patterns and their resolutions

### consolidated_*
Memories that have been consolidated to higher levels

## Feedback Types

### correct
Indicates the AI made an error that should be corrected

### prefer
States a preference for future behavior

### pattern
Explicitly teaches a new pattern

## Error Codes

### ValueError
Invalid parameters or configuration

### PermissionError
Cannot access file or directory

### TimeoutError
Operation exceeded time limit

### ConfigurationError
Invalid configuration detected
