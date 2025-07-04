# Implementation Roadmap for AI Memory System

## Quick Start MVP (Weekend Project)

### Day 1: Basic Infrastructure
```bash
# Directory structure
mkdir -p .ai/{memory,scripts,patterns}
mkdir -p .claude/hooks

# Install dependencies
pip install sqlite3 pyyaml click rich
```

### Core Scripts Needed

#### 1. Interaction Logger (2 hours)
```python
# .ai/scripts/log_interaction.py
#!/usr/bin/env python3
import sqlite3
import json
import sys
from datetime import datetime

def log_interaction(event_type, tool_name, params):
    conn = sqlite3.connect('.ai/memory/interactions.db')
    c = conn.cursor()

    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS interactions
                 (timestamp TEXT, event_type TEXT, tool_name TEXT,
                  params TEXT, session_id TEXT)''')

    # Log the interaction
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), event_type, tool_name,
               json.dumps(params), get_session_id()))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    log_interaction(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
```

#### 2. Pattern Detector (3 hours)
```python
# .ai/scripts/detect_patterns.py
#!/usr/bin/env python3
import sqlite3
import yaml
from collections import defaultdict

class SimplePatternDetector:
    def __init__(self):
        self.conn = sqlite3.connect('.ai/memory/interactions.db')

    def detect_common_sequences(self, window_size=5):
        # Get recent interactions
        c = self.conn.cursor()
        c.execute("""SELECT tool_name, params FROM interactions
                     ORDER BY timestamp DESC LIMIT 100""")

        interactions = c.fetchall()

        # Find repeated sequences
        sequences = defaultdict(int)
        for i in range(len(interactions) - window_size):
            sequence = tuple(interactions[i:i+window_size])
            sequences[sequence] += 1

        # Save patterns that appear more than twice
        patterns = []
        for seq, count in sequences.items():
            if count > 2:
                patterns.append({
                    'sequence': [s[0] for s in seq],
                    'frequency': count,
                    'confidence': count / len(interactions)
                })

        return patterns
```

#### 3. Feedback Handler (2 hours)
```python
# .ai/scripts/ai-feedback
#!/usr/bin/env python3
import click
import json
import sqlite3
from datetime import datetime

@click.command()
@click.argument('feedback_type', type=click.Choice(['correct', 'prefer', 'pattern']))
@click.argument('message')
def feedback(feedback_type, message):
    """Record developer feedback for AI learning."""

    conn = sqlite3.connect('.ai/memory/interactions.db')
    c = conn.cursor()

    # Create feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (timestamp TEXT, type TEXT, message TEXT, context TEXT)''')

    # Get recent context
    c.execute("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 5")
    recent_context = c.fetchall()

    # Store feedback
    c.execute("INSERT INTO feedback VALUES (?, ?, ?, ?)",
              (datetime.now().isoformat(), feedback_type, message,
               json.dumps(recent_context)))

    conn.commit()
    conn.close()

    click.echo(f"✓ Feedback recorded: {feedback_type} - {message}")

if __name__ == '__main__':
    feedback()
```

### Hook Configuration

#### 1. PreToolUse Hook
```bash
# .claude/hooks/pre-tool-use.sh
#!/bin/bash

# Log the interaction
python3 .ai/scripts/log_interaction.py "pre-tool" "$CLAUDE_TOOL_NAME" "$CLAUDE_TOOL_PARAMS"

# Check for relevant patterns
PATTERNS=$(python3 .ai/scripts/check_patterns.py "$CLAUDE_TOOL_NAME")
if [ ! -z "$PATTERNS" ]; then
    echo "[Memory] Similar pattern detected: $PATTERNS" >&2
fi
```

#### 2. Stop Hook
```bash
# .claude/hooks/stop.sh
#!/bin/bash

# Detect new patterns from session
python3 .ai/scripts/detect_patterns.py

# Update CLAUDE.md with learnings
python3 .ai/scripts/update_claude_md.py
```

### Claude Code Settings
```json
{
  "hooks": {
    "preToolUse": ".claude/hooks/pre-tool-use.sh",
    "postToolUse": ".claude/hooks/post-tool-use.sh",
    "stop": ".claude/hooks/stop.sh"
  }
}
```

## Minimal Viable Features

### Week 1 Deliverables

1. **Basic Logging**
   - Track all file reads/edits
   - Record command executions
   - Store in SQLite

2. **Simple Pattern Detection**
   - Identify repeated file access patterns
   - Detect common command sequences
   - Flag recurring error-fix patterns

3. **Feedback Commands**
   ```bash
   ai-feedback correct "Don't modify this file"
   ai-feedback prefer "Use async/await here"
   ai-feedback pattern "Always run tests after API changes"
   ```

4. **Auto-Generated Context**
   - Update CLAUDE.md with:
     - Frequently accessed files
     - Common workflows
     - Developer preferences

### Example Usage Flow

```bash
# Developer starts working
$ claude "help me fix the authentication bug"

# Claude reads files, memory system logs patterns
[Memory] You previously fixed similar auth issues in auth.service.ts

# Developer provides feedback
$ ai-feedback pattern "auth bugs usually need token refresh"

# Next session, Claude has context
$ claude "implement password reset"
[Memory] Based on auth patterns, checking: auth.service.ts, token.utils.ts

# Check what AI has learned
$ ai-memory show
Recent Patterns:
- Auth workflows: Read auth.service → Check tests → Update middleware
- Testing pattern: Run unit tests before integration tests
- Code style: Prefer early returns in validation
```

## Success Metrics for MVP

1. **Pattern Detection Rate**: >3 patterns identified per day
2. **Context Relevance**: >70% of suggested files are actually used
3. **Feedback Integration**: Preferences applied within 2 interactions
4. **Time Saved**: 20% reduction in context-setting time

## Next Steps After MVP

1. **Semantic Search**: Add embeddings for better pattern matching
2. **Proactive Suggestions**: "You might want to update the tests too"
3. **Team Knowledge Sharing**: Export/import memory between developers
4. **Visual Memory Browser**: Web UI to explore learned patterns

## Getting Started Today

```bash
# 1. Clone this repo
git clone <repo>

# 2. Run setup script
./setup-ai-memory.sh

# 3. Configure Claude Code
claude settings

# 4. Start using with memory
claude "let's implement that new feature"

# 5. Provide feedback as you work
ai-feedback prefer "use TypeScript strict mode"
```

The MVP can be built in a weekend and immediately start providing value by reducing repetitive context-setting.
