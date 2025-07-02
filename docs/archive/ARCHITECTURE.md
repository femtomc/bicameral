# Persistent AI Collaborator Architecture

## Overview

This document outlines the architecture for a persistent memory and learning system for Claude Code, enabling it to accumulate context and improve its effectiveness on specific codebases over time.

## Core Concept

Transform Claude Code from a stateless assistant into a persistent collaborator that:
1. Remembers interactions across sessions
2. Learns project-specific patterns automatically
3. Accepts and incorporates developer feedback
4. Improves its effectiveness over time

## MVP Scope

The Minimum Viable Product focuses on:
1. **Interaction Logging** - Track all file operations and code changes
2. **Pattern Detection** - Identify recurring workflows and conventions
3. **Feedback System** - Allow developers to correct and guide the AI
4. **Context Building** - Automatically maintain relevant project knowledge
5. **Memory Persistence** - Store learnings between sessions

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code                          │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   PreTool    │  │  PostTool    │  │    Stop      │ │
│  │    Hooks     │  │    Hooks     │  │    Hook      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼──────────────────┼──────────────────┼────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                 Memory Manager (Python)                  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Interaction │  │   Pattern    │  │   Context    │ │
│  │   Logger     │  │  Detector    │  │   Builder    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Feedback   │  │   Memory     │  │    Query     │ │
│  │   Processor  │  │ Consolidator │  │   Engine     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                    Storage Layer                         │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  .ai/memory/ │  │.ai/patterns/ │  │.ai/feedback/ │ │
│  │  SQLite DB   │  │ YAML Files   │  │ JSON Logs    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Hook Integration

#### PreToolUse Hook
```bash
#!/bin/bash
# .claude/hooks/pre-tool-use.sh

# Log the intended action
python3 .ai/scripts/log_interaction.py \
  --event "pre-tool" \
  --tool "$CLAUDE_TOOL_NAME" \
  --params "$CLAUDE_TOOL_PARAMS"

# Check if this matches a known pattern
PATTERN=$(python3 .ai/scripts/check_patterns.py \
  --tool "$CLAUDE_TOOL_NAME" \
  --params "$CLAUDE_TOOL_PARAMS")

if [ ! -z "$PATTERN" ]; then
  # Inject pattern knowledge into context
  echo "::claude-context:: Known pattern detected: $PATTERN"
fi
```

#### PostToolUse Hook
```bash
#!/bin/bash
# .claude/hooks/post-tool-use.sh

# Log results and update patterns
python3 .ai/scripts/update_memory.py \
  --event "post-tool" \
  --tool "$CLAUDE_TOOL_NAME" \
  --result "$CLAUDE_TOOL_RESULT" \
  --success "$CLAUDE_TOOL_SUCCESS"
```

#### Stop Hook
```bash
#!/bin/bash
# .claude/hooks/stop.sh

# Consolidate session memory
python3 .ai/scripts/consolidate_session.py

# Generate session summary for next time
python3 .ai/scripts/update_claude_md.py
```

### 2. Memory Manager Components

#### Interaction Logger
Records all tool usage with context:
```python
# .ai/scripts/log_interaction.py
class InteractionLogger:
    def __init__(self):
        self.db = sqlite3.connect('.ai/memory/interactions.db')
        
    def log(self, event_type, tool, params, context=None):
        timestamp = datetime.now()
        session_id = self.get_current_session()
        
        # Extract relevant info based on tool type
        if tool == "Read":
            self.log_file_read(params['file_path'], context)
        elif tool == "Edit":
            self.log_code_change(params['file_path'], 
                               params['old_string'], 
                               params['new_string'])
        # ... etc
```

#### Pattern Detector
Identifies recurring workflows:
```python
# .ai/scripts/pattern_detector.py
class PatternDetector:
    def __init__(self):
        self.patterns = self.load_patterns()
        
    def detect_workflow(self, recent_actions):
        # Look for sequences like:
        # 1. Read test file → Read implementation → Edit implementation → Run tests
        # 2. Read error → Search for similar → Apply fix pattern
        
        for pattern in self.patterns:
            if pattern.matches(recent_actions):
                return pattern
                
    def learn_new_pattern(self, actions, outcome):
        if outcome.was_successful:
            pattern = self.extract_pattern(actions)
            self.patterns.append(pattern)
            self.save_patterns()
```

#### Feedback Processor
Handles explicit developer feedback:
```python
# .ai/scripts/feedback.py
class FeedbackProcessor:
    def process_feedback(self, feedback_type, context):
        if feedback_type == "correction":
            self.update_pattern_confidence(context['pattern_id'], -0.1)
        elif feedback_type == "confirmation":
            self.update_pattern_confidence(context['pattern_id'], 0.1)
        elif feedback_type == "preference":
            self.store_preference(context['category'], context['choice'])
```

### 3. Developer Feedback Interface

#### Command-line feedback tool
```bash
# ai-feedback command
ai-feedback correct "Don't use async here - this module uses callbacks"
ai-feedback prefer "Use named exports instead of default exports"
ai-feedback pattern "When updating API endpoints, always update the OpenAPI spec"
```

#### Inline feedback in code
```python
# Developer can add comments that get picked up
def process_order(order_id):
    # @ai-feedback: Always validate order_id format before processing
    # @ai-pattern: Orders must be locked during processing
    pass
```

### 4. Storage Schema

#### SQLite Database Schema
```sql
-- .ai/memory/interactions.db

CREATE TABLE interactions (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    session_id TEXT,
    tool_name TEXT,
    file_path TEXT,
    action_type TEXT,
    details JSON
);

CREATE TABLE patterns (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT,
    trigger_sequence JSON,
    success_rate REAL,
    last_used DATETIME,
    confidence REAL
);

CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    type TEXT,
    context JSON,
    message TEXT
);

CREATE TABLE project_knowledge (
    key TEXT PRIMARY KEY,
    value JSON,
    confidence REAL,
    last_updated DATETIME
);
```

#### Pattern Storage Format
```yaml
# .ai/patterns/workflows.yaml
patterns:
  - id: "test-driven-bugfix"
    name: "TDD Bug Fix Flow"
    description: "Test-driven debugging workflow"
    triggers:
      - tool: "Read"
        file_pattern: "*.test.{js,ts}"
      - tool: "Bash"
        command_pattern: "npm test"
    sequence:
      - read_test_file
      - identify_failure
      - read_implementation
      - edit_implementation
      - run_tests
    confidence: 0.85
    success_rate: 0.92

  - id: "add-api-endpoint"
    name: "API Endpoint Addition"
    description: "Standard flow for adding new API endpoints"
    sequence:
      - read_router_file
      - add_route_definition
      - create_controller_method
      - create_service_method
      - add_validation
      - write_tests
    learned_from: 12
    confidence: 0.78
```

### 5. Context Injection System

#### Dynamic CLAUDE.md Generation
```python
# .ai/scripts/update_claude_md.py
class ClaudeMdGenerator:
    def generate(self):
        patterns = self.load_high_confidence_patterns()
        preferences = self.load_developer_preferences()
        recent_work = self.get_recent_context()
        
        content = f"""# Project Assistant Knowledge

## Learned Patterns
{self.format_patterns(patterns)}

## Developer Preferences
{self.format_preferences(preferences)}

## Recent Context
{self.format_recent_work(recent_work)}

## Common Issues and Solutions
{self.format_known_issues()}
"""
        
        with open('CLAUDE.md', 'w') as f:
            f.write(content)
```

## Implementation Phases

### Phase 1: Basic Logging (Week 1)
- [ ] Set up hook infrastructure
- [ ] Implement interaction logging
- [ ] Create SQLite database
- [ ] Basic file read/write tracking

### Phase 2: Pattern Detection (Week 2)
- [ ] Implement workflow detection
- [ ] Add pattern matching algorithms
- [ ] Create pattern storage format
- [ ] Build confidence scoring

### Phase 3: Feedback System (Week 3)
- [ ] Create feedback CLI tool
- [ ] Implement feedback processing
- [ ] Add inline comment parsing
- [ ] Build preference learning

### Phase 4: Context Building (Week 4)
- [ ] Implement dynamic CLAUDE.md generation
- [ ] Create context injection via hooks
- [ ] Add memory consolidation
- [ ] Build query system for retrieving knowledge

### Phase 5: Advanced Features (Week 5+)
- [ ] Add semantic search over memory
- [ ] Implement proactive suggestions
- [ ] Create memory visualization tools
- [ ] Add team knowledge sharing

## Security Considerations

1. **Local-Only Storage**: All memory stored in project directory
2. **No Sensitive Data**: Filter out credentials, keys, etc.
3. **User Control**: Easy commands to clear/reset memory
4. **Audit Trail**: All learning can be inspected

## Developer Experience

### Initial Setup
```bash
# Initialize AI memory for project
ai-init

# This creates:
# - .ai/ directory structure
# - .claude/hooks/ configuration
# - Initial CLAUDE.md
```

### Daily Usage
```bash
# Work normally with Claude Code
claude "implement user authentication"

# Provide feedback when needed
ai-feedback correct "use bcrypt not sha256"

# Check what AI has learned
ai-memory show patterns
ai-memory show preferences
```

### Memory Management
```bash
# Export memory for team sharing
ai-memory export > team-knowledge.json

# Import team knowledge
ai-memory import team-knowledge.json

# Clear specific patterns
ai-memory forget pattern test-driven-bugfix

# Full reset
ai-memory reset --confirm
```

## Success Metrics

1. **Reduced Repetition**: Measure how often developers need to re-explain context
2. **Pattern Recognition**: Track successful pattern applications
3. **Feedback Incorporation**: Monitor how quickly AI adapts to feedback
4. **Time Savings**: Measure reduction in task completion time
5. **Developer Satisfaction**: Survey on AI effectiveness over time

## Future Enhancements

1. **Semantic Understanding**: Use embeddings for better pattern matching
2. **Multi-Developer Support**: Merge learnings from team members
3. **IDE Integration**: Visual feedback and pattern suggestions
4. **Project Templates**: Export learned patterns as project templates
5. **Cross-Project Learning**: Share generic patterns across projects

## Conclusion

This architecture provides a practical path to building AI collaborators that genuinely improve over time. By leveraging Claude Code's hook system and maintaining local project memory, we can create a system that:

- Learns without explicit teaching
- Respects developer preferences
- Improves task-specific performance
- Maintains full user control
- Scales with project complexity

The MVP focuses on the core value proposition: reducing repetitive context-setting and building project-specific intelligence that persists across sessions.