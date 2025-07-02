# AI Memory System Design

## Executive Summary

A persistent memory and learning system for AI assistants that enables them to:
- Remember interactions across sessions
- Learn project-specific patterns automatically
- Accept and incorporate developer feedback
- Improve effectiveness over time

The system is implemented as an MCP (Model Context Protocol) server, making it compatible with Claude Code and other MCP-enabled AI tools.

## Problem Statement

Current AI coding assistants suffer from "context amnesia" - they forget everything between sessions and require developers to repeatedly explain project context. This creates friction and limits the AI's ability to become a true collaborator that understands a specific codebase.

## Solution: Persistent Collaborative Intelligence

### Core Concepts

1. **Dynamic Model Construction**: Build project-specific models on-demand based on observed patterns
2. **Hierarchical Memory**: Multiple tiers of storage from active context to long-term patterns
3. **Explicit Feedback Loop**: Developers can correct mistakes and set preferences
4. **Automatic Pattern Learning**: Detect and crystallize repeated workflows

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    AI Assistant                         │
│               (Claude Code, etc.)                       │
└────────────────────┬───────────────────────────────────┘
                     │ MCP Protocol
┌────────────────────▼───────────────────────────────────┐
│                MCP Memory Server                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Resources  │  │    Tools     │  │   Commands   │ │
│  │ @patterns    │  │ log_action   │  │ /feedback    │ │
│  │ @preferences │  │ get_context  │  │ /remember    │ │
│  │ @recent      │  │ detect_pattern│ │ /forget      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │            Memory Manager Core                   │  │
│  │                                                  │  │
│  │  • Interaction Logger    • Pattern Detector     │  │
│  │  • Feedback Processor    • Context Builder      │  │
│  │  • Memory Consolidator   • Query Engine         │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│                 Storage Layer                           │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   SQLite DB  │  │ Vector Store │  │  File Cache  │ │
│  │ interactions │  │  embeddings  │  │   patterns   │ │
│  │   feedback   │  │   for search │  │  preferences │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## MCP Server Design

### Server Configuration
```json
{
  "name": "memory-server",
  "version": "1.0.0",
  "description": "Persistent memory and learning for AI assistants",
  "mcp": {
    "type": "stdio",
    "command": "python",
    "args": ["-m", "bicamrl"],
    "env": {
      "MEMORY_DB_PATH": ".ai/memory"
    }
  }
}
```

### Exposed Resources

#### @patterns
Access learned patterns from the current project:
```
@patterns                 # All patterns
@patterns/workflows       # Workflow patterns only
@patterns/recent          # Recently detected patterns
@patterns/high-confidence # Patterns with >0.8 confidence
```

#### @preferences
Developer preferences and coding style:
```
@preferences              # All preferences
@preferences/style        # Code style preferences
@preferences/workflow     # Workflow preferences
```

#### @context
Current session and project context:
```
@context/recent-files     # Recently accessed files
@context/current-task     # Current task understanding
@context/session          # Current session summary
```

### MCP Tools

```python
# bicamrl/tools.py

TOOLS = [
    {
        "name": "log_interaction",
        "description": "Log an interaction with the codebase",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "file_path": {"type": "string"},
                "details": {"type": "object"}
            }
        }
    },
    {
        "name": "detect_pattern",
        "description": "Check if current action matches known patterns",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action_sequence": {"type": "array"}
            }
        }
    },
    {
        "name": "get_relevant_context",
        "description": "Get context relevant to current task",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string"},
                "file_context": {"type": "array"}
            }
        }
    }
]
```

### Slash Commands

```python
# bicamrl/commands.py

COMMANDS = [
    {
        "name": "feedback",
        "description": "Provide feedback to improve AI understanding",
        "args": [
            {"name": "type", "choices": ["correct", "prefer", "pattern"]},
            {"name": "message", "type": "string"}
        ]
    },
    {
        "name": "remember",
        "description": "Explicitly remember something important",
        "args": [
            {"name": "key", "type": "string"},
            {"name": "value", "type": "string"}
        ]
    },
    {
        "name": "memory",
        "description": "Query memory system",
        "args": [
            {"name": "action", "choices": ["show", "search", "stats"]},
            {"name": "query", "type": "string", "optional": true}
        ]
    }
]
```

## Memory System Components

### 1. Hierarchical Memory Architecture

```python
class MemoryHierarchy:
    def __init__(self):
        self.active_context = ActiveContext(capacity=8192)      # Current task
        self.working_memory = WorkingMemory(capacity=32768)     # Recent interactions
        self.episodic_memory = EpisodicMemory()                 # Compressed experiences
        self.semantic_memory = SemanticMemory()                 # Patterns & rules
        self.procedural_memory = ProceduralMemory()             # Workflows & strategies
```

### 2. Pattern Detection Engine

```python
class PatternDetector:
    def __init__(self):
        self.sequence_analyzer = SequenceAnalyzer()
        self.workflow_detector = WorkflowDetector()
        self.style_analyzer = StyleAnalyzer()
        
    def analyze_interaction_stream(self, interactions):
        # Detect repeated sequences
        sequences = self.sequence_analyzer.find_patterns(interactions)
        
        # Identify workflows
        workflows = self.workflow_detector.extract_workflows(sequences)
        
        # Analyze coding style
        style_patterns = self.style_analyzer.analyze(interactions)
        
        return PatternSet(sequences, workflows, style_patterns)
```

### 3. Feedback Processing

```python
class FeedbackProcessor:
    def process(self, feedback_type, message, context):
        if feedback_type == "correct":
            return self.process_correction(message, context)
        elif feedback_type == "prefer":
            return self.process_preference(message, context)
        elif feedback_type == "pattern":
            return self.process_pattern(message, context)
    
    def process_correction(self, message, context):
        # Identify what was wrong
        error_pattern = self.extract_error_pattern(context)
        
        # Create negative example
        self.memory.add_negative_example(error_pattern)
        
        # Extract correct approach
        correct_pattern = self.extract_correct_pattern(message)
        self.memory.add_positive_example(correct_pattern)
```

### 4. Context Building

```python
class ContextBuilder:
    def build_relevant_context(self, task_description):
        # Get semantic matches
        semantic_matches = self.semantic_search(task_description)
        
        # Get recent related work
        recent_relevant = self.get_recent_relevant(task_description)
        
        # Get applicable patterns
        patterns = self.get_applicable_patterns(task_description)
        
        # Build context object
        return Context(
            files=semantic_matches.files,
            patterns=patterns,
            recent_work=recent_relevant,
            preferences=self.get_relevant_preferences()
        )
```

## Data Schema

### SQLite Tables

```sql
-- Core interaction log
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    action_type TEXT NOT NULL,
    file_path TEXT,
    details JSON,
    embeddings BLOB
);

-- Detected patterns
CREATE TABLE patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,
    name TEXT,
    description TEXT,
    sequence JSON NOT NULL,
    confidence REAL DEFAULT 0.5,
    frequency INTEGER DEFAULT 1,
    last_seen DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Developer feedback
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    feedback_type TEXT NOT NULL,
    message TEXT NOT NULL,
    context JSON,
    applied BOOLEAN DEFAULT FALSE,
    impact_score REAL
);

-- Preferences and settings
CREATE TABLE preferences (
    key TEXT PRIMARY KEY,
    value JSON NOT NULL,
    category TEXT,
    confidence REAL DEFAULT 0.5,
    source TEXT, -- 'explicit' or 'inferred'
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Session summaries
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    start_time DATETIME,
    end_time DATETIME,
    summary TEXT,
    files_touched JSON,
    patterns_used JSON,
    feedback_received INTEGER DEFAULT 0
);
```

## Implementation Plan

### Phase 1: MVP (Weekend)
1. **Basic MCP Server**
   - Stdio communication
   - Simple resource exposure
   - Basic logging tool

2. **Core Memory System**
   - SQLite storage
   - Interaction logging
   - Simple pattern detection

3. **Feedback Commands**
   - `/feedback` command
   - Preference storage
   - Basic retrieval

### Phase 2: Pattern Intelligence (Week 2)
1. **Advanced Pattern Detection**
   - Sequence analysis
   - Workflow extraction
   - Confidence scoring

2. **Context Building**
   - Relevance matching
   - Context summarization
   - Dynamic resource generation

### Phase 3: Semantic Understanding (Week 3)
1. **Embeddings Integration**
   - Code embeddings
   - Semantic search
   - Similar pattern matching

2. **Proactive Assistance**
   - Pattern suggestions
   - Workflow automation
   - Predictive context

## Usage Examples

### Initial Setup
```bash
# Install the memory server
pip install ai-memory-server

# Configure with Claude Code
claude mcp add memory-server "python -m bicamrl"

# Or configure manually in settings
```

### Daily Workflow

```
You: Help me implement user authentication

Claude: I'll help with authentication. Based on @patterns/auth, you typically:
1. Use JWT tokens with 15-minute expiry
2. Store refresh tokens in Redis
3. Use bcrypt for password hashing

Should I follow this pattern?

You: Yes, but use 30-minute expiry this time

Claude: Understood. I'll use 30-minute expiry. Let me update that preference.

[Uses tool: log_interaction to record this preference]

You: /feedback prefer "JWT expiry: 30 minutes for this project"

Claude: ✓ Preference recorded. I'll use 30-minute JWT expiry going forward.
```

### Checking Memory

```
You: /memory show patterns

Memory Server: 
📊 Learned Patterns:
• API Endpoint Creation (confidence: 0.92)
  - Create route → Add controller → Add service → Write tests
  
• Test-Driven Bugfix (confidence: 0.87)
  - Read failing test → Understand issue → Fix implementation → Verify

• Database Migration Flow (confidence: 0.95)
  - Create migration → Update models → Run migration → Test
```

## Benefits

1. **Reduced Context Setup**: No need to explain project structure repeatedly
2. **Consistent Patterns**: AI follows established project conventions
3. **Learning from Mistakes**: Corrections improve future interactions
4. **Team Knowledge Sharing**: Export/import memory across team
5. **Tool Agnostic**: Works with any MCP-compatible AI tool

## Security & Privacy

1. **Local Storage**: All data stored in project directory
2. **No Cloud Sync**: Unless explicitly configured
3. **Sensitive Data Filtering**: Automatic filtering of credentials/keys
4. **Access Control**: Respects file system permissions
5. **Audit Trail**: All interactions logged and queryable

## Success Metrics

1. **Context Efficiency**: Time saved on context setup (target: 50% reduction)
2. **Pattern Recognition**: Successful pattern applications (target: 80% accuracy)
3. **Feedback Impact**: Behavior changes from feedback (target: 90% incorporation)
4. **Developer Satisfaction**: Survey scores (target: 4.5/5)

## Future Enhancements

1. **Multi-Project Learning**: Share common patterns across projects
2. **Team Collaboration**: Merge team members' learned patterns
3. **IDE Integration**: Visual pattern suggestions in editor
4. **Learning Analytics**: Dashboard showing AI improvement over time
5. **Pattern Marketplace**: Share useful patterns with community

## Conclusion

This design provides a practical path to building AI assistants that genuinely improve over time through:
- Persistent memory across sessions
- Automatic pattern detection
- Developer feedback incorporation
- MCP protocol for tool compatibility

The system balances automation with developer control, enabling AI assistants to become true collaborators that understand and adapt to specific codebases and team preferences.