# Bicamrl

A persistent memory and behavior system for AI assistants using MCP (Model Context Protocol).

**Works with any MCP-compatible AI client** - Claude Desktop, custom implementations, or future MCP adopters.

## Overview

Bicamrl provides AI assistants with:
- **Persistent Memory**: Stores interactions, patterns, and preferences across sessions
- **Pattern Detection**: Automatically discovers recurring workflows
- **Context Retrieval**: Provides relevant past interactions and files
- **Feedback Learning**: Adapts based on corrections and preferences

## Installation

### Prerequisites

- Python 3.9+
- [Pixi](https://pixi.sh) package manager (recommended)
- Any MCP-compatible AI client

### Setup

```bash
# Clone the repository
git clone https://github.com/femtomc/bicamrl.git
cd bicamrl

# Install with Pixi (recommended)
pixi install

# Or install with pip
pip install -e .
```

## Configuration

### Quick Start

For Claude Desktop users:

```bash
claude mcp add bicamrl "python -m bicamrl"
```

For other MCP clients, add to your configuration:

```json
{
  "mcpServers": {
    "bicamrl": {
      "command": "python",
      "args": ["-m", "bicamrl.server"],
      "env": {
        "MEMORY_DB_PATH": ".bicamrl/memory"
      }
    }
  }
}
```

Bicamrl will automatically create a `.bicamrl` directory in your home folder.

### Advanced Configuration (Optional)

Create `~/.bicamrl/Mind.toml` for additional features:

```toml
# ~/.bicamrl/Mind.toml
[sleep]
enabled = true
batch_size = 10
analysis_interval = 300
min_confidence = 0.7

[sleep.llm_providers.openai]
api_key = "${OPENAI_API_KEY}"  # Or paste your key directly
model = "gpt-4-turbo-preview"
max_tokens = 4096
```

## Features

### Core Memory System
- **Interaction Tracking**: Records queries, actions, and outcomes
- **Pattern Detection**: Identifies recurring workflows using fuzzy matching
- **Preference Storage**: Remembers coding style and tool preferences
- **Context Search**: Finds relevant past interactions
- **Memory Consolidation**: Automatically archives old data to maintain performance

### Available Resources

Query these resources to understand what Bicamrl has learned:

- `@bicamrl/patterns` - Detected patterns and workflows
- `@bicamrl/preferences` - Your coding style preferences
- `@bicamrl/context/recent` - Recent interactions and files
- `@bicamrl/sleep/insights` - Analysis insights (requires Sleep config)
- `@bicamrl/sleep/status` - System status

### Available Tools

#### Memory Operations
- `start_interaction` - Begin tracking a new interaction
- `log_action` - Record an action within an interaction
- `complete_interaction` - Finish and analyze an interaction
- `search_memory` - Find relevant past interactions
- `get_memory_stats` - View memory statistics
- `consolidate_memories` - Manually trigger memory cleanup

#### Pattern & Context
- `detect_pattern` - Check if actions match known patterns
- `get_relevant_context` - Retrieve task-specific information

#### Feedback
- `record_feedback` - Store corrections and preferences

## How It Works

### Example: Learning File Locations

```
Day 1:
User: "Fix the authentication bug"
AI: *searches many files, eventually finds bug in src/auth/token.js*

Day 3:
User: "There's another auth bug"
AI: *checks src/auth/token.js first based on previous pattern*
Time saved: 25 seconds
```

### Example: Learning Preferences

```
User: "Add user registration"
AI: "Should I use JWT with RS256?"
User: "No, we use HS256 in this project"
AI: *records preference*

Next time:
AI: *automatically uses HS256 for JWT tokens*
```

## Development

### Running Tests

```bash
pixi run test              # All tests
pixi run test-cov          # Coverage report
```

### Code Quality

```bash
pixi run check             # Run all checks
pixi run format            # Format code
pixi run lint              # Run linter
```

## License

MIT License - see [LICENSE](LICENSE) for details
