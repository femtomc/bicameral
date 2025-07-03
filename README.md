# Bicamrl

> **⚠️ EXPERIMENTAL: DO NOT USE YET!**
> 
> This project is under active development and is not ready for production use.
> APIs, data formats, and behaviors may change without notice.

A persistent memory and behavior system for AI assistants using MCP (Model Context Protocol).

**Works with any MCP-compatible AI client** - Claude Desktop, custom implementations, or future MCP adopters.

## Overview

Bicamrl provides AI assistants with:
- **Dynamic World Modeling**: Builds understanding of any domain through LLM-powered inference
- **Persistent Memory**: Stores interactions and learned world models across sessions
- **Adaptive Learning**: Continuously refines understanding based on new interactions
- **Context Retrieval**: Provides relevant past interactions and domain knowledge
- **Feedback Integration**: Updates world models based on corrections and outcomes

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

# For cloud-based LLMs
[sleep.llm_providers.openai]
api_key = "${OPENAI_API_KEY}"  # Or paste your key directly
model = "gpt-4-turbo-preview"
max_tokens = 4096

# For local LLMs via LM Studio
[sleep.llm_providers.lmstudio]
api_base = "http://localhost:1234/v1"
model = "your-local-model"
```

## Features

### Core Memory System
- **World Model Construction**: Builds dynamic understanding of any domain (coding, cooking, physics, music, etc.)
- **Interaction Tracking**: Records queries, actions, and outcomes to refine models
- **Model Persistence**: Saves learned world models across sessions
- **Context-Aware Retrieval**: Provides relevant knowledge based on current task
- **Memory Consolidation**: Hierarchical memory system that promotes important knowledge

### Available Resources

Query these resources to understand what Bicamrl has learned:

- `@bicamrl/world_models` - Domain understanding and behavioral models
- `@bicamrl/preferences` - Learned preferences and approaches
- `@bicamrl/context/recent` - Recent interactions and relevant knowledge
- `@bicamrl/sleep/insights` - Deep analysis insights (requires Sleep config)
- `@bicamrl/sleep/status` - System status and learning progress

### Available Tools

#### Memory Operations
- `start_interaction` - Begin tracking a new interaction
- `log_action` - Record an action within an interaction
- `complete_interaction` - Finish and analyze an interaction
- `search_memory` - Find relevant past interactions
- `get_memory_stats` - View memory statistics
- `consolidate_memories` - Manually trigger memory cleanup

#### World Model & Context
- `analyze_domain` - Build understanding of new domains
- `get_relevant_context` - Retrieve domain-specific knowledge

#### Feedback
- `record_feedback` - Store corrections and preferences

## How It Works

### Dynamic World Modeling

Bicamrl uses LLM inference to build understanding of any domain without hardcoded assumptions:

```
User: "Help me debug this quantum circuit"
Bicamrl: *analyzes interaction, builds quantum computing world model*
- Understands qubit states, gate operations, measurement
- Learns common debugging approaches for quantum circuits
- Adapts responses based on quantum physics principles
```

### Example: Cross-Domain Learning

```
Session 1 - Cooking:
User: "My soufflé keeps collapsing"
Bicamrl: *builds culinary world model*
- Understands temperature sensitivity, protein structures
- Learns timing and technique relationships

Session 2 - Music:
User: "This chord progression sounds muddy"
Bicamrl: *builds music theory world model*
- Understands harmonic relationships, frequency overlap
- Applies knowledge to suggest voicing changes
```

### Persistent Learning

World models persist across sessions, continuously improving:

```
Week 1: Basic understanding of your codebase structure
Week 2: Knows your testing patterns and error handling style
Week 3: Anticipates common refactoring needs in your domain
```

## Development

### Local Testing with LM Studio

For development without API keys:

1. Install and run [LM Studio](https://lmstudio.ai/)
2. Load a model (e.g., Mistral, Llama)
3. Configure in `~/.bicamrl/Mind.toml` (see Advanced Configuration)
4. Run tests with: `pixi run test-lmstudio "your-model-name"`

### Running Tests

```bash
pixi run test              # All tests
pixi run test-cov          # Coverage report
pixi run test-lmstudio     # Test with local LLM
```

### Code Quality

```bash
pixi run check             # Run all checks
pixi run format            # Format code
pixi run lint              # Run linter
```

## License

MIT License - see [LICENSE](LICENSE) for details
