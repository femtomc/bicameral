# Bicamrl Design Document

## Executive Summary

Bicamrl is a two-tier architecture for AI assistants that separates immediate task execution from meta-cognitive observation and planning. The system provides persistent memory, pattern detection, and preference learning, with a roadmap toward probabilistic modeling and planning.

## Table of Contents

1. [Vision](#vision)
2. [Architecture Overview](#architecture-overview)
3. [Current Implementation](#current-implementation)
4. [Evolution Roadmap](#evolution-roadmap)
5. [Technical Design](#technical-design)
6. [Usage Examples](#usage-examples)
7. [Research Foundations](#research-foundations)
8. [Implementation Plan](#implementation-plan)

## Vision

Bicamrl explores a two-tier architecture for AI systems:

1. **Central Mind**: Executes immediate tasks using available tools
2. **Meta-Cognitive Layer**: Observes interactions, detects patterns, and maintains long-term memory

The core hypothesis is that separating execution from observation/planning enables better adaptation to user preferences and workflow patterns over time.

## Architecture Overview

### Two-Tier System

```
┌─────────────────────────────────────────────────────────────────┐
│                     META-COGNITIVE AGENT                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  World Model    │  │   Planner    │  │  Model Learner  │   │
│  │                 │  │              │  │                 │   │
│  │ • Transitions   │  │ • MCTS       │  │ • Bayesian      │   │
│  │ • Rewards       │  │ • Beam Search│  │ • Online        │   │
│  │ • Preferences   │  │ • GenJAX     │  │ • Active        │   │
│  └────────┬────────┘  └──────┬───────┘  └────────┬────────┘   │
│           └───────────────────┴──────────────────┴─────────┐   │
│                                                            ▼   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Planning Engine                        │  │
│  │  "Given current state and goal, what sequence of         │  │
│  │   high-level actions maximizes expected reward?"         │  │
│  └────────────────────────┬─────────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            │ PLAN DISPATCH
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CENTRAL MIND AGENT                          │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Action Executor │  │   MCP Tools  │  │ User Interface  │   │
│  │                 │  │              │  │                 │   │
│  │ • Interpret     │  │ • File ops   │  │ • Chat          │   │
│  │ • Implement     │  │ • Code gen   │  │ • Feedback      │   │
│  │ • Verify        │  │ • Testing    │  │ • Clarify       │   │
│  └────────┬────────┘  └──────┬───────┘  └────────┬────────┘   │
└───────────────────────┼─────────────────┼─────────────────────┘
                        │                  │
                        │ EXECUTION REPORT │
                        └──────────────────┘
```

### Memory Architecture

```
┌─────────────────────────────────────────┐
│          HIERARCHICAL MEMORY            │
├─────────────────────────────────────────┤
│  ACTIVE (< 1 hour)                      │
│  • Current task context                 │
│  • Recent interactions                  │
├─────────────────────────────────────────┤
│  WORKING (< 24 hours)                   │
│  • Today's patterns                     │
│  • Session preferences                  │
├─────────────────────────────────────────┤
│  EPISODIC (< 7 days)                    │
│  • Recent workflows                     │
│  • Project-specific patterns            │
├─────────────────────────────────────────┤
│  SEMANTIC (Permanent)                   │
│  • Learned patterns                     │
│  • Developer preferences                │
│  • Probabilistic world model            │
└─────────────────────────────────────────┘
```

## Current Implementation

### Core Components

1. **MCP Server** (`server.py`)
   - FastMCP implementation
   - Exposes tools and resources via Model Context Protocol

2. **Memory Manager** (`core/memory.py`)
   - SQLite-based persistent storage
   - Hierarchical memory organization (active, working, episodic, semantic)

3. **Pattern Detector** (`core/pattern_detector.py`)
   - Identifies repeated action sequences
   - Calculates confidence based on frequency and recency

4. **Feedback Processor** (`core/feedback_processor.py`)
   - Stores user preferences from explicit feedback
   - Updates stored patterns based on outcomes

### MCP Resources

- `@bicamrl/patterns` - Learned workflow patterns
- `@bicamrl/preferences` - Developer preferences
- `@bicamrl/context/recent` - Recent activity
- `@bicamrl/kbm/insights` - Meta-cognitive insights

### MCP Tools

- `log_interaction` - Record development actions
- `detect_pattern` - Find matching patterns
- `get_relevant_context` - Retrieve task context
- `record_feedback` - Store developer feedback
- `search_memory` - Query stored information
- `optimize_prompt` - Enhance prompts (Sleep Layer)

## Evolution Roadmap

### Phase 1: Current MVP ✓
- Basic memory persistence
- Pattern detection
- Feedback processing
- MCP integration

### Phase 2: Knowledge Base Maintainer
- Multi-LLM coordination
- Pattern analysis
- Prompt optimization
- Meta-cognitive insights

### Phase 3: Probabilistic Enhancement
- GenJAX integration
- Uncertainty quantification
- Bayesian preference learning
- Probabilistic pattern models

### Phase 4: Model-Based Planning
- World model of development
- Monte Carlo planning
- Strategic action sequences
- Outcome prediction

### Phase 5: Full Meta-Cognition
- Self-steering capabilities
- Custom planner generation
- Active learning
- Multi-objective optimization

## Technical Design

### Data Schemas

#### Interaction
```python
@dataclass
class Interaction:
    timestamp: datetime
    action: str  # "edit_file", "run_test", etc.
    file_path: Optional[str]
    details: Dict[str, Any]
    context_before: str
    context_after: str
    success: bool
    user_feedback: Optional[str]
```

#### Pattern
```python
@dataclass
class Pattern:
    name: str
    description: str
    action_sequence: List[str]
    frequency: int
    success_rate: float
    average_duration: float
    context_triggers: List[str]
    learned_at: datetime
```

#### World Model State
```python
@dataclass
class WorkflowState:
    """High-level state representation for any workflow."""
    domain: str  # "software", "research", "writing", "design", etc.
    task_progress: Dict[str, float]  # {"draft": 0.8, "research": 0.3}
    quality_metrics: Dict[str, float]  # Domain-specific quality measures
    user_satisfaction: float
    recent_context: List[str]
```

#### Workflow Action
```python
@dataclass
class WorkflowAction:
    """High-level action representation for any domain."""
    action_type: str  # "research", "create", "revise", "analyze", etc.
    focus_area: str  # What specifically to work on
    approach: Dict[str, Any]  # How to do it
    intensity: float  # 0-1 effort level
```

### Probabilistic Models

#### World Model (GenJAX)
```python
@gen.gen_fn
def workflow_model(goal, initial_state, horizon):
    """Generative model of any interactive workflow."""
    
    # User preferences
    preferences = gen.sample("preferences", preference_prior())
    
    # Task dynamics (how complexity evolves)
    complexity_growth = gen.sample("complexity", gen.gamma(2.0, 0.5))
    
    trajectory = []
    state = initial_state
    
    for t in range(horizon):
        # Action selection based on domain
        action = gen.sample(f"action_{t}", 
            policy(state, goal, preferences))
        
        # State transition
        next_state = gen.sample(f"state_{t+1}", 
            transition_kernel(state, action))
        
        # User feedback
        feedback = gen.sample(f"feedback_{t}",
            feedback_model(next_state, preferences))
        
        trajectory.append((action, next_state, feedback))
        state = next_state
    
    return trajectory
```

### Communication Protocol

#### Plan Dispatch
```python
@dataclass
class PlanDispatch:
    plan_id: str
    actions: List[WorkflowAction]
    expected_outcomes: List[ExpectedOutcome]
    confidence: float
    rationale: str
```

#### Execution Report
```python
@dataclass
class ExecutionReport:
    plan_id: str
    executed_actions: List[WorkflowAction]
    observed_states: List[WorkflowState]
    user_feedback: List[UserFeedback]
    success_metrics: Dict[str, float]
```

## Usage Examples

### Current System

```python
# AI Assistant logs interaction
await log_interaction(
    action="implement_auth",
    file_path="app/auth.py",
    details={"method": "JWT", "library": "python-jose"}
)

# System detects pattern
patterns = await detect_pattern(["setup_project", "implement_auth"])
# Returns: "TDD workflow detected with 85% confidence"

# Developer provides feedback
await record_feedback(
    feedback_type="prefer",
    message="use RS256 algorithm for JWT tokens"
)

# Future interactions adapt
context = await get_relevant_context("implement JWT")
# Returns: Previous implementations, preferences, patterns
```

### Future System (Model-Based Planning)

```python
# The meta-cognitive agent maintains a probabilistic model of workflows
# and uses it to plan sequences of actions

# Example: Software task
plan = await meta_cognitive.plan(
    goal="Add authentication to API",
    current_state=WorkflowState(
        domain="software",
        task_progress={"api": 0.8, "auth": 0.0},
        quality_metrics={"tests": 0.6, "security": 0.0}
    )
)

# Returns a plan with uncertainty quantification:
PlanDispatch(
    actions=[
        WorkflowAction("implement", "jwt_middleware", {"library": "jose"}, 0.8),
        WorkflowAction("test", "auth_endpoints", {"coverage": "full"}, 0.9),
        WorkflowAction("document", "auth_flow", {"format": "openapi"}, 0.6)
    ],
    confidence=0.75,
    rationale="Based on 12 similar tasks, JWT implementation typically precedes testing"
)

# The system learns from execution outcomes
report = ExecutionReport(
    executed_actions=plan.actions[:2],  # User stopped after tests
    user_feedback=["prefer bcrypt over jose for this project"],
    success_metrics={"tests_passing": 1.0, "time_taken": 45}
)

# Model updates its beliefs about this user's preferences
await meta_cognitive.update_model(plan, report)
```

## Research Foundations

The design draws from:

1. **Building Machines that Learn and Think with People** (Collins et al., 2024)
   - Argument for hybrid cognitive architectures

2. **Self-Steering Language Models** (Grand et al., 2025)
   - arXiv:2504.07081
   - Language models generating task-specific inference programs

3. **Syntactic and Semantic Control of LLMs via SMC** (Loula et al., 2025)
   - arXiv:2504.13139
   - Sequential Monte Carlo for constrained generation

4. **PoE-World** (Piriyakulkij et al., 2025)
   - Compositional approach to world modeling

## Implementation Plan

### MVP Target (Phase 1-2)

**Goal**: Functional memory and pattern detection system

**Components**:
1. MCP server exposing tools and resources
2. Persistent hierarchical memory
3. Pattern detection from repeated sequences
4. Preference storage from user feedback
5. Basic Knowledge Base Maintainer

### Future System (Phase 3-5)

**Research Goals**:
1. Probabilistic modeling of user workflows
2. Planning algorithms for action sequences
3. Uncertainty quantification in predictions
4. Active learning from limited feedback

## Key Design Principles

1. **Separation of Concerns**: Execution (central) vs. observation/planning (meta-cognitive)
2. **Natural Language Abstraction**: States and actions represented as high-level concepts
3. **Persistent Memory**: Interactions stored hierarchically by time and relevance
4. **Pattern Detection**: Identifies repeated sequences in user workflows
5. **Preference Learning**: Adapts based on explicit feedback
6. **Modular Architecture**: Components can be developed and tested independently

## Security & Privacy

- All data stored locally
- No telemetry without consent
- Encrypted preference storage
- Audit logs for compliance
- Configurable data retention

## Performance Considerations

- Async operations throughout
- Efficient SQLite queries
- Lazy pattern matching
- Incremental learning
- JAX compilation for inference

## Future Directions

- Probabilistic world models for better uncertainty handling
- Planning algorithms that reason multiple steps ahead
- Active learning to request feedback when uncertain
- Model-based reasoning about action outcomes