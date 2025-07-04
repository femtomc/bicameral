# Bicamrl Project Instructions

**IMPORTANT**: This file contains critical project guidelines. Read it before making any changes.

## System Capabilities

Bicamrl provides:
- **Hybrid Storage**: SQLite + vector embeddings for interaction tracking
- **TOML Configuration**: All settings via Mind.toml
- **LM Studio Integration**: Full support for local LLMs
- **LLM-Based Role Discovery**: Dynamic analysis of interaction patterns
- **Markdown Role Storage**: Human-readable role definitions
- **Production Logging**: Structured JSON logging with MCP decorators
- **Count-Based Memory Consolidation**: Interaction count thresholds
- **Archival System**: Automatic cleanup with interaction_id tracking
- **Dynamic World Modeling**: LLM-driven discovery without hardcoded patterns
- **World Model Persistence**: SQLite storage for discovered knowledge
- **Optimized JSON Prompts**: Streamlined for instruction-tuned models

### Key Architectural Decisions

**IMPORTANT**: These are core design choices that must be maintained:

- **Configuration**: Mind.toml for all settings (roles in `[roles]` section)
- **Role Storage**: `~/.bicamrl/roles/` with both .md and .json formats
- **Communication**: Wake-Sleep via MCP tools and shared storage
- **Processing**: All operations are async
- **Memory Consolidation**: COUNT-BASED thresholds (NOT time-based)
- **Memory Hierarchy**: Active → Working → Episodic → Semantic
- **Data Management**: Archive tables for cleaned up data
- **World Modeling**: Dynamic LLM-driven discovery (NO hardcoded patterns)
- **World Persistence**: Dedicated SQLite database for world knowledge

## Common Commands

```bash
# Development
pixi run server          # Start MCP server
pixi run test            # Run test suite (ALWAYS run before commits)
pixi run check           # Format, lint, type check, and test
pixi run test-cov        # Coverage report

# Testing Modes
pixi run test-production         # Production tests
pixi run test-lmstudio "model"   # Test with LM Studio
pixi run test-mcp                # MCP-specific tests

# Code Quality
pixi run format          # Auto-format code with Ruff and Cargo
pixi run format-check    # Check formatting without modifying files
pixi run lint            # Run linter with Ruff and Clippy
pixi run type-check      # Type check with Pyright and Cargo

# Utilities
pixi run demo            # Interactive demo

# Rust-specific
pixi run cargo-check     # Check Rust code compilation
pixi run cargo-clippy    # Run Rust linter (clippy)
pixi run cargo-test      # Run Rust tests
pixi run cargo-fmt-check # Check Rust formatting

# Pre-commit setup
pre-commit install       # Install git hooks (run once)
pre-commit run --all     # Run pre-commit on all files
```

## CRITICAL DIRECTIVES

**YOU MUST**:
1. **NO NEW DOCUMENTS**: Do not create new documentation files unless explicitly requested. Update existing docs or add code comments instead.
2. **NO EMOJIS**: Use clear, blunt, precise communication. No emojis in code or documentation.
3. **READ DOCS FIRST**: Always read the docs in `docs/` directory before making changes.
4. **REMOVE OLD CODE**: We do NOT care about backwards compatibility. Remove legacy code freely.
5. **TEST BEFORE COMMITTING**: Run `pixi run test` before any commits.
6. **USE COUNT-BASED CONSOLIDATION**: Memory consolidation is based on interaction counts, NOT time.
7. **TESTS GO IN TESTS/**: NEVER create test files in examples/ or scripts/. All tests MUST go in the tests/ directory. Examples are for user-facing demos only.
8. **NO TEST SCRIPTS**: NEVER create standalone test scripts like test_*.py in the root directory. ALL tests go in tests/ directory as proper pytest tests.

## Terminology & Naming Conventions

**YOU MUST** use these exact terms:
- "Sleep" (NOT "SleepLayer")
- "Wake" (NOT "WakeLayer")
- "Memory" (NOT "MemoryManager")
- "consolidation" (NOT "promotion" or "migration")
- These represent the two-tier architecture

## Getting Started

**IMPORTANT**: Before making ANY changes, read these documents in order:

1. **`docs/ARCHITECTURE.md`** - Technical reference (START HERE)
2. **`docs/DESIGN.md`** - Design philosophy
3. **`docs/TODO.md`** - Current implementation status
4. **`docs/getting-started.md`** - User guide
5. **`docs/api-reference.md`** - API documentation

## Code Style

- **Formatting**: Ruff (run `pixi run format`)
- **Linting**: Ruff (run `pixi run lint`)
- **Type Checking**: Pyright (run `pixi run type-check`)
- **Imports**: Use relative imports within modules
- **Docstrings**: Required for all public functions
- **Tests**: Required for all new features

## Project Overview

Bicamrl is a MCP (Model Context Protocol) server providing persistent memory for AI assistants. It implements a hierarchical memory system inspired by cognitive science principles.

## Architecture

### Core Components

1. **Memory System** (`bicamrl/core/memory.py`)
   - Hierarchical memory: Active → Working → Episodic → Semantic
   - Count-based consolidation thresholds
   - SQLite backend with archive tables
   - Optional HybridStore integration for vector search

2. **Pattern Detection** (`bicamrl/core/pattern_detector.py`)
   - Identifies recurring workflows and action sequences
   - Confidence scoring for pattern reliability
   - Fully dynamic discovery without hardcoded patterns

3. **Feedback Processing** (`bicamrl/core/feedback_processor.py`)
   - Handles developer corrections and preferences
   - Updates pattern confidence based on feedback

4. **Memory Consolidation** (`bicamrl/core/memory_consolidator.py`)
   - Count-based thresholds (10 interactions → working memory)
   - LLM-powered semantic extraction
   - Automatic archival of consolidated memories

5. **Sleep** (`bicamrl/sleep/`)
   - Meta-cognitive layer for continuous improvement
   - Multi-LLM coordination for different tasks
   - Prompt optimization based on learned patterns

6. **World Model** (`bicamrl/core/world_model.py`)
   - Dynamic entity and domain discovery via LLM
   - Persistent storage of discovered world knowledge
   - No hardcoded domains, patterns, or entity types

### MCP Implementation

The server uses FastMCP from the official Python SDK:
- Entry point: `bicamrl/__main__.py`
- Server implementation: `bicamrl/server.py`
- Tools and resources exposed via decorators
- Async lifecycle management

## Developer Workflow

1. **Research First**: Use `Grep` and `Glob` to understand existing code
2. **Test-Driven**: Write or update tests before implementation
3. **Incremental Changes**: Make small, focused commits
4. **Verify Changes**: Always run `pixi run test` before moving on

## Repository Etiquette

- **Commit Messages**: Clear, concise descriptions of changes
- **No Force Push**: Never force push to shared branches
- **Clean History**: Squash commits when appropriate
- **Documentation**: Update docs when changing functionality

## Key Design Decisions

1. **Modular Architecture**: Core functionality separate from MCP layer
2. **Async First**: All I/O operations are async
3. **Type Safety**: Comprehensive type hints throughout
4. **Testability**: Mock-friendly design for unit testing

## Testing Status

All systems tested and functional:
- MCP server with FastMCP
- Hierarchical memory system with count-based consolidation
- Pattern detection with confidence scoring
- Sleep system with multi-LLM support
- Test suite with integration and stress tests
- Documentation in `docs/` directory
- Archive system for memory management
- Interaction tracking with unique IDs
- Dynamic world modeling via LLM
- World model persistence in SQLite
- LM Studio support for local testing

## Implementation Specifics

### Memory Consolidation Thresholds

**IMPORTANT**: These are the exact thresholds used:

```python
self.active_to_working_threshold = 10   # 10 interactions → working memory
self.working_to_episodic_threshold = 5  # 5 working memories → episodic
self.episodic_to_semantic_threshold = 10 # 10 episodes → semantic
self.min_frequency_for_semantic = 5     # Pattern frequency for semantic
```

### Database Schema

**Key Tables**:
- `interactions` - Raw interaction data with interaction_id
- `complete_interactions` - Full conversation cycles
- `patterns` - Detected patterns and consolidated memories
- `archived_interactions` - Cleaned up interactions
- `archived_patterns` - Promoted patterns

**World Model Tables**:
- `world_model_states` - Complete world model snapshots with entities, relations, goals
- `world_model_snapshots` - Historical snapshots of world model evolution

**Archive Reasons**:
- `consolidated_to_working` - Raw interactions → working memory
- `consolidated_to_episodic` - Working → episodic memory
- `promoted_to_semantic` - Pattern → semantic knowledge

## Testing & Verification

**IMPORTANT**: Always verify changes with:

```bash
pixi run test              # Must pass before ANY commit
pixi run test-cov          # Check coverage for new code
pixi run check             # Format, lint, type check, and test
```

### LLM Testing Modes

Tests can run in three modes:
1. **Mock Mode (default)**: No external dependencies, uses mock responses
2. **LM Studio Mode**: Uses local LLM via LM Studio
3. **OpenAI Mode**: Uses OpenAI API (requires API key)

```bash
# Run with mock LLM (default, fastest)
pixi run test

# Run tests with LM Studio (requires model specified)
pixi run test-lmstudio "your-model-name"

# Run specific test with LM Studio
pixi run python -m pytest tests/test_world_model.py -m lmstudio -v

# Test LM Studio connection
pixi run python -m pytest tests/test_lmstudio_connection.py -v -s
```

**Important Notes**:
- Tests default to mock mode for reliability and speed
- Only tests explicitly marked with `@pytest.mark.lmstudio` will use LM Studio
- LM Studio must be running on localhost:1234 for tests to work
- Mock responses are optimized for instruction-tuned models

## Environment Setup

### Required Environment Variables

```bash
# For Sleep system (optional)
export OPENAI_API_KEY="your-key"  # Or use LM Studio

# For custom paths (optional)
export MEMORY_DB_PATH=".bicamrl/memory"
```

### LM Studio Configuration

```toml
# In ~/.bicamrl/Mind.toml
[sleep.llm_providers.lmstudio]
api_base = "http://localhost:1234/v1"
model = "your-local-model"
```

## Key Files Reference

**IMPORTANT**: When working on specific features, start with these files:

### Memory System
- `bicamrl/core/memory.py` - Main memory class
- `bicamrl/core/memory_consolidator.py` - Consolidation logic
- `bicamrl/storage/sqlite_store.py` - Database operations

### Pattern Detection
- `bicamrl/core/pattern_detector.py` - Pattern matching
- `bicamrl/core/interaction_pattern_detector.py` - NLP-based detection

### MCP Server
- `bicamrl/server.py` - Server implementation
- `bicamrl/__main__.py` - Entry point

### Sleep System
- `bicamrl/sleep/sleep.py` - Background processing (not sleep_layer.py!)
- `bicamrl/sleep/roles.py` - Behavioral roles

### World Model
- `bicamrl/core/world_model.py` - Dynamic world modeling with LLM inference
- `bicamrl/core/llm_service.py` - Centralized LLM service for all AI operations
- `bicamrl/storage/sqlite_store.py` - World model persistence (see add_world_model_state)

### Logging Infrastructure
- `bicamrl/utils/logging_config.py` - Structured JSON logging setup
- `bicamrl/utils/log_utils.py` - Logging utilities and context managers
- `bicamrl/utils/mcp_logging.py` - MCP-specific logging decorators

## Logging Best Practices

### Using MCP Decorators

All MCP tools and resources should use the logging decorators:

```python
from .utils.mcp_logging import mcp_tool_logger, mcp_resource_logger

@mcp.tool()
@mcp_tool_logger("my_tool")
async def my_tool(param1: str) -> Dict[str, Any]:
    # Decorator automatically logs:
    # - Tool invocation with parameters
    # - Execution time
    # - Results or errors
    # - Session context
    return {"result": "success"}
```

### Logging Metrics

Use `log_tool_metric` for important measurements:

```python
from .utils.mcp_logging import log_tool_metric

# Inside your tool
patterns_found = len(results)
log_tool_metric(
    "patterns_detected",
    patterns_found,
    "my_tool",
    confidence=0.85
)
```

## Common Pitfalls

- **Memory Consolidation**: Uses COUNT-based thresholds, not time
- **Async Operations**: All I/O must be async
- **Import Errors**: Run from project root with `python -m bicamrl`
- **Test Failures**: Check if LLM providers are configured
- **Database Migrations**: Handled automatically in SQLiteStore
- **Logging Verbosity**: Use decorators instead of manual logging in tools
- **World Model**: Never hardcode domains or patterns - let LLM discover them
- **JSON Prompts**: Keep prompts concise for instruction-tuned models
- **LM Studio**: Ensure it's running on localhost:1234 before testing
