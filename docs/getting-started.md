# Getting Started with Bicamrl

Bicamrl gives AI assistants persistent memory and the ability to learn from interactions.

## What Bicamrl Does

- **Remembers across sessions**: Past interactions, preferences, and patterns persist
- **Learns your workflows**: Automatically detects repeated sequences of actions
- **Adapts to your style**: Applies learned preferences to future interactions
- **Improves over time**: Optional meta-cognitive analysis identifies optimization opportunities

## Prerequisites

- Python 3.9+
- Any MCP-compatible client:
  - Claude Desktop (Anthropic)
  - Other MCP clients (see [MCP ecosystem](https://modelcontextprotocol.io))
  - Custom MCP implementations

## Installation

```bash
# Install from PyPI (when available)
pip install bicamrl

# Or install from source
git clone https://github.com/femtomc/bicamrl.git
cd bicamrl
pip install -e .
```

## Configuration

### For Claude Desktop

Edit the configuration file:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "bicamrl": {
      "command": "python",
      "args": ["-m", "bicamrl.server"],
      "env": {}
    }
  }
}
```

### For Other MCP Clients

Bicamrl runs as a standard MCP server. Configure it according to your client's documentation:

```bash
# Start server directly
python -m bicamrl.server

# Or with custom settings
MEMORY_DB_PATH=/custom/path python -m bicamrl.server
```

Bicamrl will create a `.bicamrl` directory in your home folder to store memory.

## First Use

Once configured, Bicamrl works automatically. Try these examples:

### Check what Claude has learned
```
"What patterns have you noticed in my work?"
"What are my coding preferences?"
```

### Teach preferences
```
"I always use pytest for testing"
"I prefer async/await over callbacks"
```

### Get contextual help
```
"Based on my usual workflow, how should I structure this feature?"
"What files do I typically modify together?"
```

## How Memory Works

Bicamrl automatically:
1. **Logs interactions** - Every file access, edit, and command
2. **Detects patterns** - Sequences repeated 3+ times become patterns
3. **Learns preferences** - Explicit feedback shapes future behavior
4. **Consolidates knowledge** - Important patterns become permanent

## Available Commands

### MCP Tools
- `log_interaction` - Record specific actions
- `detect_pattern` - Check if a sequence matches known patterns
- `record_feedback` - Store preferences and corrections
- `search_memory` - Find relevant past interactions
- `get_memory_insights` - Get context-specific recommendations

### MCP Resources
- `@bicamrl/patterns` - View all learned patterns
- `@bicamrl/preferences` - See stored preferences
- `@bicamrl/context/recent` - Recent activity summary

## Common Workflows

### Teaching a new pattern
Simply repeat an action sequence 3+ times. Bicamrl will automatically detect and remember it.

### Correcting behavior
```
"That's not right. I always run linting before tests."
"Don't use global variables in this project."
```

### Building context
The more you work with Claude + Bicamrl, the better it understands your style, preferences, and project patterns.

## Advanced Features (Optional)

### Enable Sleep Layer (Background Processing)

For meta-cognitive analysis, create `bicamrl_config.json`:

```json
{
  "kbm": {
    "enabled": true,
    "llm_providers": {
      "openai": {
        "api_key": "YOUR_API_KEY"
      }
    }
  }
}
```

This enables:
- Automatic insight generation
- Pattern optimization recommendations
- Proactive performance improvements

## Troubleshooting

### Bicamrl not working?
1. Check Claude Desktop logs for errors
2. Verify Python path in config matches your system
3. Ensure `.bicamrl` directory has write permissions

### Memory not persisting?
- Bicamrl creates `.bicamrl` in your home directory
- Check that the directory exists and contains `memory.db`

## Privacy

All data is stored locally in `~/.bicamrl`. Nothing is sent to external servers unless you enable Sleep Layer with API keys.
