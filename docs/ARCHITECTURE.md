# Bicamrl Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architectural Principles](#architectural-principles)
3. [System Components](#system-components)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Memory Architecture](#memory-architecture)
6. [Pattern Detection System](#pattern-detection-system)
7. [Sleep](#sleep)
8. [MCP Integration](#mcp-integration)
9. [Storage Architecture](#storage-architecture)
10. [Security Architecture](#security-architecture)
11. [Performance Architecture](#performance-architecture)
12. [Deployment Architecture](#deployment-architecture)
13. [Future Architecture Evolution](#future-architecture-evolution)

## System Overview

Bicamrl implements a two-tier cognitive architecture for AI assistants:

```
┌─────────────────────────────────────────────────────────────────┐
│                     META-COGNITIVE LAYER                        │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐     │
│  │      Sleep      │  │   Pattern    │  │    Learning     │     │
│  │                 │  │   Analyzer   │  │    Engine       │     │
│  └────────┬────────┘  └───────┬──────┘  └────────┬────────┘     │
│           └───────────────────┴──────────────────┴─────────┐    │
│                                                            ▼    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Observation & Analysis                │   │
│  └────────────────────────┬─────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │ INSIGHTS & PATTERNS
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                           │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ Memory Manager  │  │  MCP Server  │  │ Tool Providers  │    │
│  │                 │  │              │  │                 │    │
│  │ • Hierarchical  │  │ • Resources  │  │ • log_interact  │    │
│  │ • Persistent    │  │ • Tools      │  │ • detect_pattern│    │
│  │ • Searchable    │  │ • Lifecycle  │  │ • get_context   │    │
│  └────────┬────────┘  └──────┬───────┘  └────────┬────────┘    │
└───────────┼──────────────────┼───────────────────┼─────────────┘
            └──────────────────┴───────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │   Storage    │
                        │   (SQLite)   │
                        └──────────────┘
```

## Architectural Principles

### 1. Separation of Concerns
- **Execution Layer**: Handles immediate tasks and interactions
- **Meta-cognitive Layer**: Observes, learns, and optimizes over time
- **Storage Layer**: Manages persistent data with clear interfaces

### 2. Hierarchical Memory Model
Based on cognitive science principles:
- **Active Memory** (< 1 hour): Immediate context
- **Working Memory** (< 24 hours): Session patterns
- **Episodic Memory** (< 7 days): Recent workflows
- **Semantic Memory** (Permanent): Learned patterns and preferences

### 3. Event-Driven Architecture
- Asynchronous operations throughout
- Event-based communication between components
- Non-blocking I/O for scalability

### 4. Plugin Architecture
- MCP protocol for extensibility
- Modular component design
- Clear interfaces between layers

### 5. Privacy-First Design
- Local-only storage by default
- No telemetry without consent
- User-controlled data lifecycle

## System Components

### Core Components

#### 1. MCP Server (`server.py`)
```python
class BicameralServer:
    """FastMCP server implementation with lifecycle management."""

    def __init__(self):
        self.server = Server("bicamrl")
        self.memory_manager = MemoryManager()
        self.pattern_detector = PatternDetector()
        self.feedback_processor = FeedbackProcessor()
        self.sleep_layer = SleepLayer()  # Optional
```

**Responsibilities:**
- MCP protocol implementation
- Tool and resource registration
- Lifecycle management
- Component coordination

#### 2. Memory Manager (`core/memory.py`)
```python
class MemoryManager:
    """Hierarchical memory system with automatic consolidation."""

    async def store_interaction(self, interaction: Interaction)
    async def search_memory(self, query: str, memory_type: Optional[str])
    async def get_relevant_context(self, task_description: str)
    async def consolidate_memories()
```

**Key Features:**
- SQLite-based persistence
- Automatic memory promotion
- Relevance-based retrieval
- Time-decay weighting

#### 3. Pattern Detector (`core/pattern_detector.py`)
```python
class PatternDetector:
    """Identifies recurring action sequences with fuzzy matching."""

    async def detect_pattern(self, action_sequence: List[str])
    async def calculate_similarity(self, seq1: List[str], seq2: List[str])
    async def update_pattern_confidence(self, pattern_id: str, outcome: bool)
```

**Algorithms:**
- Levenshtein distance for fuzzy matching
- Time-based confidence decay
- Minimum occurrence threshold
- Contextual pattern matching

#### 4. Feedback Processor (`core/feedback_processor.py`)
```python
class FeedbackProcessor:
    """Processes and stores user feedback and preferences."""

    async def record_feedback(self, feedback_type: str, content: str)
    async def get_preferences(self, category: Optional[str])
    async def resolve_conflicts(self, preferences: List[Preference])
```

**Capabilities:**
- Preference categorization
- Conflict resolution
- Inheritance hierarchy (global → project → file)

### Optional Components

#### 5. Sleep (`sleep/sleep_layer.py`)
```python
class SleepLayer:
    """Meta-cognitive analysis and optimization engine."""

    async def observe_interaction(self, interaction: Dict)
    async def analyze_batch(self, observations: List[Dict])
    async def generate_insights()
    async def optimize_prompts(self, context: Dict)
```

**Features:**
- Multi-LLM coordination
- Pattern mining
- Prompt optimization
- Performance analysis

## Data Flow Architecture

### Interaction Flow
```
User Action → MCP Client → MCP Server → Tool Execution
                                     ↓
                            Memory Manager ← Interaction Data
                                     ↓
                            Pattern Detector ← Action Sequence
                                     ↓
                            Sleep (if enabled) ← Observations
                                     ↓
                            Storage Layer ← Persistent Data
```

### Retrieval Flow
```
User Query → MCP Client → MCP Server → Resource Handler
                                    ↓
                           Memory Search ← Query Parameters
                                    ↓
                           Relevance Scoring ← Results
                                    ↓
                           Context Building ← Ranked Results
                                    ↓
                           Response → MCP Client → User
```

## Memory Architecture

### Memory Types and Transitions

```
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│   ACTIVE    │  10    │   WORKING   │   5    │  EPISODIC   │
│  (Recent)   │ ────→  │  (Session)  │ ────→  │  (Weekly)   │
└─────────────┘  int.  └─────────────┘  mem.  └─────────────┘
                                                      │
                                                      │ High
                                                      │ Value
                                                      ↓
                                              ┌─────────────┐
                                              │  SEMANTIC   │
                                              │ (Permanent) │
                                              └─────────────┘
```

### Memory Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT CHECK(memory_type IN ('active', 'working', 'episodic', 'semantic')),
    content TEXT NOT NULL,
    embedding BLOB,  -- For future semantic search
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    importance_score REAL DEFAULT 0.0,
    decay_rate REAL DEFAULT 1.0
);

CREATE INDEX idx_memory_type ON memories(memory_type);
CREATE INDEX idx_created_at ON memories(created_at);
CREATE INDEX idx_importance ON memories(importance_score DESC);
```

### Consolidation Algorithm

```python
async def consolidate_memories(self):
    """Promote valuable memories to higher tiers."""

    # Active → Working (after 1 hour)
    active_memories = await self.get_old_memories('active', hours=1)
    for memory in active_memories:
        if memory.access_count > 2 or memory.importance_score > 0.7:
            await self.promote_memory(memory, 'working')

    # Working → Episodic (after 1 day)
    working_memories = await self.get_old_memories('working', days=1)
    for memory in working_memories:
        # Calculate composite score
        score = self.calculate_memory_score(memory)
        if score > 0.6:
            await self.promote_memory(memory, 'episodic')

    # Episodic → Semantic (after 7 days)
    episodic_memories = await self.get_old_memories('episodic', days=7)
    for memory in episodic_memories:
        if memory.access_count > 10 or memory.importance_score > 0.8:
            await self.promote_memory(memory, 'semantic')

def calculate_memory_score(self, memory) -> float:
    """Calculate composite importance score for memory."""
    # Base score from explicit importance
    base_score = memory.importance_score

    # Access frequency bonus
    access_bonus = min(memory.access_count / 10, 0.3)

    # Recency bonus (exponential decay)
    age_days = (datetime.now() - memory.created_at).days
    recency_bonus = 0.2 * math.exp(-age_days / 7)

    # Pattern association bonus
    pattern_bonus = 0.1 if memory.pattern_associations else 0

    return min(base_score + access_bonus + recency_bonus + pattern_bonus, 1.0)
```

### Memory Search and Ranking

```python
async def search_memory(self, query: str, memory_types: List[str] = None) -> List[Memory]:
    """Search memories with relevance ranking."""

    # Get candidate memories
    candidates = await self.store.search_memories(
        query=query,
        memory_types=memory_types or ['semantic', 'episodic', 'working']
    )

    # Calculate relevance scores
    scored_results = []
    for memory in candidates:
        score = self.calculate_relevance(query, memory)
        scored_results.append((score, memory))

    # Sort by relevance
    scored_results.sort(key=lambda x: x[0], reverse=True)

    return [memory for _, memory in scored_results[:20]]

def calculate_relevance(self, query: str, memory: Memory) -> float:
    """Calculate relevance score for search results."""

    # Text similarity (simple token overlap for now)
    query_tokens = set(query.lower().split())
    memory_tokens = set(memory.content.lower().split())
    text_similarity = len(query_tokens & memory_tokens) / max(len(query_tokens), 1)

    # Type weight (prefer higher-tier memories)
    type_weights = {
        'semantic': 1.0,
        'episodic': 0.8,
        'working': 0.6,
        'active': 0.4
    }
    type_weight = type_weights.get(memory.memory_type, 0.5)

    # Recency factor
    age_days = (datetime.now() - memory.accessed_at).days
    recency_factor = math.exp(-age_days / 30)

    # Importance factor
    importance_factor = memory.importance_score

    # Composite score
    return (
        0.4 * text_similarity +
        0.2 * type_weight +
        0.2 * recency_factor +
        0.2 * importance_factor
    )
```

## Pattern Detection System

### Pattern Types

1. **File Access Patterns** - Files frequently accessed together
2. **Action Sequence Patterns** - Repeated sequences of actions
3. **Workflow Patterns** - Higher-level task workflows
4. **Error Patterns** - Common error sequences and fixes
5. **Consolidated Patterns** - Patterns promoted from lower tiers

### Pattern Matching Algorithm

```python
def calculate_similarity(self, seq1: List[str], seq2: List[str]) -> float:
    """Calculate similarity using normalized Levenshtein distance."""

    if not seq1 or not seq2:
        return 0.0

    # Normalize sequences
    seq1_normalized = [self.normalize_action(a) for a in seq1]
    seq2_normalized = [self.normalize_action(a) for a in seq2]

    # Calculate Levenshtein distance
    distance = levenshtein_distance(seq1_normalized, seq2_normalized)
    max_length = max(len(seq1), len(seq2))

    # Convert to similarity score (0-1)
    similarity = 1 - (distance / max_length)

    # Apply time decay
    time_factor = self.calculate_time_decay(seq2.timestamp)

    return similarity * time_factor

def normalize_action(self, action: str) -> str:
    """Normalize action strings for comparison."""
    # Remove common prefixes
    prefixes = ['get_', 'set_', 'update_', 'create_', 'delete_']
    for prefix in prefixes:
        if action.startswith(prefix):
            action = action[len(prefix):]

    # Convert to lowercase and remove special characters
    action = action.lower().replace('_', '').replace('-', '')

    return action

def calculate_time_decay(self, timestamp: str) -> float:
    """Calculate exponential decay based on recency."""
    now = datetime.now()
    action_time = datetime.fromisoformat(timestamp)
    days_ago = (now - action_time).days

    # Exponential decay with 7-day half-life
    decay_rate = 0.693 / self.recency_weight_days  # ln(2) / half-life
    return math.exp(-decay_rate * days_ago)
```

### Fuzzy Matching Implementation

```python
def levenshtein_distance(self, s1: List[str], s2: List[str]) -> int:
    """Calculate Levenshtein distance between sequences."""
    m, n = len(s1), len(s2)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    return dp[m][n]

def find_fuzzy_matches(self, target_seq: List[str], threshold: float = 0.7) -> List[Pattern]:
    """Find patterns that fuzzy match the target sequence."""
    matches = []

    for pattern in self.all_patterns:
        similarity = self.calculate_similarity(target_seq, pattern.sequence)
        if similarity >= threshold:
            matches.append({
                'pattern': pattern,
                'similarity': similarity,
                'match_type': 'fuzzy' if similarity < 1.0 else 'exact'
            })

    return sorted(matches, key=lambda x: x['similarity'], reverse=True)
```

### Pattern Storage Schema

```sql
CREATE TABLE patterns (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    action_sequence JSON NOT NULL,
    trigger_conditions JSON,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    confidence_score REAL DEFAULT 0.5,
    last_matched TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

CREATE INDEX idx_confidence ON patterns(confidence_score DESC);
CREATE INDEX idx_last_matched ON patterns(last_matched DESC);
```

### Workflow Detection Algorithm

```python
async def _detect_workflow_patterns(self, interactions: List[Dict]) -> List[Dict]:
    """Detect higher-level workflow patterns."""
    patterns = []

    # Sort interactions by timestamp (oldest first)
    sorted_interactions = sorted(interactions, key=lambda x: x['timestamp'])

    # Group interactions by time proximity (within 5 minutes)
    workflows = []
    current_workflow = []
    last_time = None

    for interaction in sorted_interactions:
        timestamp = datetime.fromisoformat(interaction['timestamp'])

        if last_time and (timestamp - last_time) > timedelta(minutes=5):
            # Time gap - end current workflow
            if len(current_workflow) >= self.min_frequency:
                workflows.append(current_workflow)
            current_workflow = []

        current_workflow.append(interaction)
        last_time = timestamp

    # Process final workflow
    if len(current_workflow) >= self.min_frequency:
        workflows.append(current_workflow)

    # Analyze workflows for patterns
    workflow_signatures = []
    for workflow in workflows:
        # Create workflow signature
        signature = self._create_workflow_signature(workflow)
        workflow_signatures.append(signature)

    # Find repeated workflow patterns
    signature_counts = Counter(workflow_signatures)
    for signature, count in signature_counts.items():
        if count >= 2:  # Workflow repeated at least twice
            patterns.append({
                "name": f"Workflow: {signature}",
                "description": "Multi-step workflow pattern",
                "pattern_type": "workflow",
                "sequence": signature.split(" → "),
                "frequency": count,
                "confidence": min(count / len(workflows), 1.0)
            })

    return patterns

def _create_workflow_signature(self, workflow: List[Dict]) -> str:
    """Create a normalized signature for a workflow."""
    # Extract key actions and files
    actions = []
    files = set()

    for interaction in workflow:
        action = interaction.get('action', '')
        file_path = interaction.get('file_path')

        # Normalize action
        if action in ['edit_file', 'save_file', 'write_file']:
            action = 'modify'
        elif action in ['read_file', 'open_file', 'view_file']:
            action = 'read'

        actions.append(action)
        if file_path:
            # Extract file extension as part of signature
            ext = file_path.split('.')[-1] if '.' in file_path else 'unknown'
            files.add(ext)

    # Create signature
    action_summary = " → ".join(actions[:5])  # Limit to first 5 actions
    file_summary = f"[{','.join(sorted(files))}]" if files else ""

    return f"{action_summary} {file_summary}".strip()
```

### Pattern Confidence Update

```python
def update_confidence(self, pattern_id: str, success: bool):
    """Update pattern confidence using Bayesian approach."""

    pattern = self.get_pattern(pattern_id)

    # Prior confidence
    prior = pattern.confidence_score

    # Likelihood of success given pattern
    if success:
        likelihood = 0.8  # P(success|pattern)
    else:
        likelihood = 0.2  # P(failure|pattern)

    # Bayesian update
    posterior = (likelihood * prior) / (
        likelihood * prior + (1 - likelihood) * (1 - prior)
    )

    # Apply learning rate
    learning_rate = 0.1
    new_confidence = prior + learning_rate * (posterior - prior)

    # Update pattern metadata
    pattern.confidence_score = new_confidence
    pattern.last_matched = datetime.now()
    if success:
        pattern.success_count += 1
    else:
        pattern.failure_count += 1

    pattern.save()
```

## Complete Interaction Model

### Overview

The system now tracks complete conversation cycles from user query to feedback, enabling true learning from natural language patterns.

### Interaction Lifecycle

```python
@dataclass
class Interaction:
    """Complete interaction cycle between user and AI."""

    # Identity
    interaction_id: str
    session_id: str
    timestamp: datetime

    # User Input
    user_query: str                      # Natural language request
    query_context: Dict[str, Any]        # Current files, recent actions

    # AI Processing
    ai_interpretation: Optional[str]     # What AI understood
    planned_actions: List[str]           # Actions AI plans to take
    confidence: float                    # AI's confidence in interpretation
    active_role: Optional[str]           # Which behavioral role was active

    # Execution
    actions_taken: List[Action]          # Actual actions performed
    execution_started_at: Optional[datetime]
    execution_completed_at: Optional[datetime]
    tokens_used: int

    # Outcome
    user_feedback: Optional[str]         # User's response
    feedback_type: FeedbackType          # APPROVAL, CORRECTION, FOLLOWUP, etc.
    success: bool                        # Whether the interaction succeeded
```

### Interaction Pattern Detection

```python
class InteractionPatternDetector:
    """Detects patterns in complete user interactions."""

    async def detect_intent_patterns(self, interactions: List[Interaction]):
        """Map user vocabulary to action sequences."""
        # Group by similar queries that led to same actions
        # Learn: "fix auth bug" → [search "auth", edit "token.js"]

    async def detect_success_patterns(self, interactions: List[Interaction]):
        """Identify what makes interactions successful."""
        # Analyze approved vs corrected interactions
        # Learn what approaches work best

    async def detect_correction_patterns(self, interactions: List[Interaction]):
        """Learn from misinterpretations."""
        # Track when AI misunderstood user intent
        # Build correction mappings
```

### Natural Language Processing

The system implements lightweight NLP for pattern matching:

```python
def calculate_query_similarity(self, query1: str, query2: str) -> float:
    """Calculate semantic similarity between queries."""
    # Extract keywords (remove stop words)
    # Calculate Jaccard similarity
    # Use embeddings if available

def extract_key_phrases(self, text: str) -> List[str]:
    """Extract important phrases from user queries."""
    # N-gram extraction
    # Frequency analysis
    # Domain-specific term detection
```

### Interaction Logging

```python
class InteractionLogger:
    """Tracks complete interaction cycles."""

    def start_interaction(self, user_query: str) -> str:
        """Begin tracking a new interaction."""

    def log_interpretation(self, interpretation: str, planned_actions: List[str]):
        """Record what the AI understood and plans to do."""

    def log_action(self, action_type: str, target: str, details: Dict):
        """Log each action as it's performed."""

    def complete_interaction(self, feedback: str = None, success: bool = None):
        """Finalize the interaction with user feedback."""
```

## Sleep

### Architecture

```
┌─────────────────────────────────────────────────┐
│              Sleep Coordinator                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │ Observation │  │   Analysis  │  │ Action  ││
│  │   Queue     │  │   Engine    │  │ Engine  ││
│  └──────┬──────┘  └──────┬──────┘  └────┬────┘│
│         ↓                 ↓              ↓     │
│  ┌─────────────────────────────────────────┐  │
│  │          LLM Provider Manager            │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐    │  │
│  │  │OpenAI  │  │Claude  │  │ Mock   │    │  │
│  │  └────────┘  └────────┘  └────────┘    │  │
│  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Multi-LLM Coordination

```python
class LLMProviderManager:
    """Manages multiple LLM providers with role assignment."""

    def __init__(self, config: Dict):
        self.providers = {
            'openai': OpenAIProvider(config['openai']),
            'claude': ClaudeProvider(config['claude']),
            'mock': MockProvider()  # For testing
        }

        self.roles = {
            'analyzer': config['roles']['analyzer'],
            'generator': config['roles']['generator'],
            'optimizer': config['roles']['optimizer']
        }

    async def analyze_patterns(self, observations: List[Dict]) -> List[Insight]:
        """Use analyzer role to extract patterns."""
        provider = self.providers[self.roles['analyzer']]
        return await provider.analyze(observations)
```

### Insight Generation Pipeline

```python
async def generate_insights(self):
    """Multi-stage insight generation pipeline."""

    # Stage 1: Pattern Mining
    patterns = await self.mine_patterns(self.observation_buffer)

    # Stage 2: Error Analysis
    errors = await self.analyze_errors(self.observation_buffer)

    # Stage 3: Performance Optimization
    optimizations = await self.find_optimizations(patterns, errors)

    # Stage 4: Knowledge Synthesis
    insights = await self.synthesize_knowledge(
        patterns, errors, optimizations
    )

    # Stage 5: Validation & Ranking
    validated_insights = await self.validate_insights(insights)

    return self.rank_insights(validated_insights)
```

## MCP Integration

### Resource Implementation

```python
@server.list_resources()
async def list_resources() -> List[Resource]:
    """Expose Bicamrl resources via MCP."""
    return [
        Resource(
            uri="bicamrl://patterns",
            name="Learned Patterns",
            description="Workflow patterns detected from usage",
            mimeType="application/json"
        ),
        Resource(
            uri="bicamrl://preferences",
            name="Developer Preferences",
            description="Coding style and tool preferences",
            mimeType="application/json"
        ),
        Resource(
            uri="bicamrl://context/recent",
            name="Recent Context",
            description="Recently accessed files and interactions",
            mimeType="application/json"
        ),
        Resource(
            uri="bicamrl://kbm/insights",
            name="Sleep Insights",
            description="Meta-cognitive analysis results",
            mimeType="application/json"
        )
    ]
```

### Tool Implementation

```python
@server.tool()
async def log_interaction(
    action: str,
    file_path: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    context_before: Optional[str] = None,
    context_after: Optional[str] = None,
    success: bool = True
) -> Dict[str, Any]:
    """Log a development interaction for learning."""

    interaction = Interaction(
        timestamp=datetime.now(),
        action=action,
        file_path=file_path,
        details=details or {},
        context_before=context_before or "",
        context_after=context_after or "",
        success=success
    )

    # Store in memory
    await memory_manager.store_interaction(interaction)

    # Detect patterns
    recent_actions = await memory_manager.get_recent_actions(10)
    patterns = await pattern_detector.detect_pattern(
        [a.action for a in recent_actions]
    )

    # Send to Sleep if enabled
    if kbm and kbm.enabled:
        await kbm.observe_interaction(interaction.to_dict())

    return {
        "status": "logged",
        "interaction_id": interaction.id,
        "detected_patterns": patterns
    }
```

## Storage Architecture

### Database Design

```sql
-- Core tables
CREATE TABLE interactions (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    action TEXT NOT NULL,
    file_path TEXT,
    details JSON,
    context_before TEXT,
    context_after TEXT,
    success BOOLEAN DEFAULT TRUE,
    user_feedback TEXT,
    session_id TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- New complete interactions table for full conversation tracking
CREATE TABLE complete_interactions (
    interaction_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    user_query TEXT NOT NULL,
    ai_interpretation TEXT,
    success BOOLEAN DEFAULT 0,
    feedback_type TEXT,
    execution_time REAL,
    tokens_used INTEGER DEFAULT 0,
    data TEXT NOT NULL  -- Full interaction JSON
);

CREATE TABLE patterns (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    action_sequence JSON NOT NULL,
    frequency INTEGER DEFAULT 1,
    success_rate REAL DEFAULT 0.0,
    average_duration REAL,
    context_triggers JSON,
    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence REAL DEFAULT 0.5
);

CREATE TABLE preferences (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSON NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT CHECK(source IN ('explicit', 'inferred')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category, key)
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    summary JSON,
    metrics JSON
);

-- Sleep tables
CREATE TABLE sleep_observations (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    interaction_id TEXT,
    observation_type TEXT,
    content JSON,
    processed BOOLEAN DEFAULT FALSE,
    batch_id TEXT,
    FOREIGN KEY (interaction_id) REFERENCES interactions(id)
);

CREATE TABLE sleep_insights (
    id TEXT PRIMARY KEY,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    insight_type TEXT,
    content JSON,
    confidence REAL,
    applied BOOLEAN DEFAULT FALSE,
    impact_metrics JSON
);

-- Indexes for performance
CREATE INDEX idx_interactions_timestamp ON interactions(timestamp DESC);
CREATE INDEX idx_interactions_action ON interactions(action);
CREATE INDEX idx_interactions_file ON interactions(file_path);
CREATE INDEX idx_patterns_confidence ON patterns(confidence DESC);
CREATE INDEX idx_preferences_category ON preferences(category);
CREATE INDEX idx_sleep_observations_batch ON sleep_observations(batch_id);
CREATE INDEX idx_complete_interactions_session ON complete_interactions(session_id);
CREATE INDEX idx_complete_interactions_timestamp ON complete_interactions(timestamp);
CREATE INDEX idx_complete_interactions_success ON complete_interactions(success);
```

### Storage Manager

```python
class SQLiteStore:
    """Async SQLite storage with connection pooling."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._pool = []
        self._pool_size = 5

    async def get_connection(self):
        """Get connection from pool or create new."""
        if self._pool:
            return self._pool.pop()
        return await aiosqlite.connect(self.db_path)

    async def return_connection(self, conn):
        """Return connection to pool."""
        if len(self._pool) < self._pool_size:
            self._pool.append(conn)
        else:
            await conn.close()

### Hybrid Storage Architecture

The system now implements a hybrid storage approach combining structured data in SQLite with vector embeddings for semantic search:

```python
class HybridStore:
    """Combines SQLite for structured data with vector storage for embeddings."""

    def __init__(self, storage_path: Path):
        self.sqlite_store = SQLiteStore(storage_path / "bicamrl.db")
        self.vector_store = VectorStore(storage_path / "vectors")

    async def add_interaction(self, interaction: Interaction):
        # Store structured data in SQLite
        await self.sqlite_store.add_complete_interaction(interaction.to_dict())

        # Generate and store embeddings for semantic search
        query_embedding = self._generate_embedding(interaction.user_query)
        await self.vector_store.add_embedding(
            embedding_id=f"query_{interaction.interaction_id}",
            embedding=query_embedding,
            metadata={...}
        )
```

### Vector Storage

```python
class VectorStore:
    """Simple vector storage with similarity search."""

    async def add_embedding(self, embedding_id: str, embedding: np.ndarray, metadata: Dict):
        """Store embedding with metadata."""

    async def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple]:
        """Find k most similar embeddings using cosine similarity."""
```
```

## Security Architecture

### Data Protection

1. **Local-Only Storage**
   - All data stored in user's filesystem
   - No cloud sync without explicit consent
   - Data never leaves the local machine

2. **Sensitive Data Filtering**
   ```python
   def filter_sensitive_data(self, content: str) -> str:
       """Remove potential sensitive information."""
       # API keys
       content = re.sub(r'[A-Za-z0-9]{32,}', '[REDACTED_KEY]', content)
       # Email addresses
       content = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', content)
       # Credit card numbers
       content = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CC]', content)
       return content
   ```

3. **Access Control**
   - File system permissions
   - Process isolation
   - No network access by default

### Privacy Controls

```python
class PrivacyManager:
    """Manages user privacy preferences and data retention."""

    async def apply_retention_policy(self):
        """Delete data according to retention settings."""
        settings = await self.get_privacy_settings()

        if settings.auto_delete_interactions:
            cutoff = datetime.now() - timedelta(days=settings.retention_days)
            await self.delete_interactions_before(cutoff)

        if settings.anonymize_old_data:
            await self.anonymize_data_before(cutoff)

    async def export_user_data(self) -> Dict:
        """Export all user data for portability."""
        return {
            'interactions': await self.get_all_interactions(),
            'patterns': await self.get_all_patterns(),
            'preferences': await self.get_all_preferences(),
            'insights': await self.get_all_insights()
        }
```

## Performance Architecture

### Optimization Strategies

1. **Async Everything**
   ```python
   async def process_request(self, request):
       # Parallel processing where possible
       results = await asyncio.gather(
           self.memory_manager.search(request.query),
           self.pattern_detector.find_matches(request.context),
           self.kbm.get_recommendations(request.task)
       )
       return self.combine_results(results)
   ```

2. **Caching Layer**
   ```python
   class CacheManager:
       def __init__(self, ttl: int = 300):  # 5 minute TTL
           self.cache = {}
           self.ttl = ttl

       async def get_or_compute(self, key: str, compute_fn):
           if key in self.cache:
               entry = self.cache[key]
               if time.time() - entry['time'] < self.ttl:
                   return entry['value']

           value = await compute_fn()
           self.cache[key] = {'value': value, 'time': time.time()}
           return value
   ```

3. **Database Optimization**
   - Connection pooling
   - Prepared statements
   - Batch operations
   - Strategic indexes

### Performance Metrics

```python
class PerformanceMonitor:
    """Tracks system performance metrics."""

    async def record_operation(self, operation: str, duration: float):
        await self.store.record_metric(
            operation=operation,
            duration=duration,
            timestamp=datetime.now()
        )

    async def get_performance_report(self) -> Dict:
        return {
            'average_response_time': await self.calculate_avg_response(),
            'pattern_detection_accuracy': await self.calculate_accuracy(),
            'memory_usage': await self.get_memory_stats(),
            'cache_hit_rate': await self.calculate_cache_hits(),
            'error_rate': await self.calculate_error_rate()
        }
```

## Deployment Architecture

### Directory Structure
```
$HOME/.bicamrl/
├── memory.db          # SQLite database
├── config.json        # User configuration
├── logs/              # Application logs
│   └── bicamrl.log
├── backups/           # Automatic backups
│   └── memory-2024-01-01.db
└── exports/           # Data exports
    └── insights-2024-01.json
```

### Configuration Management

```python
class ConfigManager:
    """Manages configuration with environment variable overrides."""

    def load_config(self) -> Dict:
        # 1. Load defaults
        config = self.get_defaults()

        # 2. Load from file
        if self.config_file.exists():
            file_config = json.loads(self.config_file.read_text())
            config = deep_merge(config, file_config)

        # 3. Apply environment overrides
        for key, value in os.environ.items():
            if key.startswith('BICAMERAL_'):
                self.apply_env_override(config, key, value)

        # 4. Validate
        self.validate_config(config)

        return config
```

### Lifecycle Management

```python
class LifecycleManager:
    """Manages application lifecycle."""

    async def startup(self):
        """Initialize all components."""
        await self.init_database()
        await self.load_configuration()
        await self.start_background_tasks()
        await self.warm_caches()

    async def shutdown(self):
        """Graceful shutdown."""
        await self.stop_background_tasks()
        await self.flush_pending_writes()
        await self.close_connections()
        await self.create_backup()
```

## Future Architecture Evolution

### Phase 3: Probabilistic Enhancement
```
┌─────────────────────────────────────────────────┐
│           Probabilistic Layer (GenJAX)           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │  Bayesian   │  │   Pattern   │  │  User   │ │
│  │  Preference │  │   Models    │  │ Models  │ │
│  │   Learning  │  │             │  │         │ │
│  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────┘
```

### Phase 4: Model-Based Planning
```
┌─────────────────────────────────────────────────┐
│              Planning Engine                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │World Model  │  │   Planner   │  │Outcome  │ │
│  │            │  │   (MCTS)    │  │Predictor│ │
│  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────┘
```

### Phase 5: Full Meta-Cognition
```
┌─────────────────────────────────────────────────┐
│          Self-Steering Architecture              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   Custom    │  │   Active    │  │  Multi- │ │
│  │  Planner    │  │  Learning   │  │Objective│ │
│  │ Generation  │  │   Agent     │  │Optimizer│ │
│  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────┘
```

## Conclusion

The Bicamrl architecture provides a robust foundation for building AI assistants with persistent memory and meta-cognitive capabilities. The modular design allows for incremental enhancement while maintaining system stability and performance. The clear separation between execution and observation layers enables sophisticated learning without impacting real-time performance.

Key architectural benefits:
- **Modularity**: Components can be developed and tested independently
- **Extensibility**: New capabilities can be added without breaking existing functionality
- **Scalability**: Async architecture handles increasing workloads efficiently
- **Privacy**: Local-first design respects user data sovereignty
- **Evolution**: Clear path from simple pattern detection to full meta-cognition
