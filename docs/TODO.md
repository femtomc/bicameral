# Bicamrl Implementation TODO

## Overview

This document tracks the sequence of tasks required to implement the MVP as described in DESIGN.md. The MVP target includes Phase 1 (current implementation improvements) and Phase 2 (basic Knowledge Base Maintainer).

## Current Status

### ✅ Completed
- FastMCP server implementation
- Hierarchical memory system with SQLite
- Pattern detector with fuzzy matching and time-based weighting
- Feedback processor
- MCP tools and resources
- Comprehensive test suite (integration, stress, performance)
- Sleep fully integrated with server
- Command role system for behavioral templates
- Role discovery and management
- Prompt optimization with role context
- Complete documentation suite

### 🚧 In Progress
- Production hardening
- Multi-LLM provider testing
- Performance optimization
- Writing tests for role system
- Updating documentation with role concepts

### ❌ Not Started
- Automatic interaction completion on session end
- Advanced feedback processing features
- A/B testing for prompts
- Cost tracking for LLM providers

### ✅ Just Completed (This Session)
- **Completed transition to interaction-based API**:
  - Updated all tests to use new API (start_interaction → log_action → complete_interaction)
  - Fixed MCP compliance tests for FastMCP API
  - Updated MCP integration tests with correct interaction flow
  - Fixed stress performance tests to use InteractionLogger
  - Removed all uses of old log_interaction API
- **Fixed critical bugs**:
  - Fixed IndexError in prompt_optimizer.py when files list is empty
  - Fixed Sleep config key from "sleep" to "sleep_layer"
  - Fixed pattern_detector wrapper to return dict properly
  - Fixed resource reading to handle FastMCP's Resource objects
  - Added missing FeedbackType import in hybrid_store.py
- **Enhanced Role Discovery System**:
  - Removed all pre-defined/builtin roles
  - Created InteractionRoleDiscoverer for discovering roles from complete interactions
  - Implemented sophisticated pattern analysis using query patterns, action sequences, and success metrics
  - Added support for both sklearn-based and simple clustering
  - Integrated role discovery with hybrid store for semantic analysis
  - Added `discover_roles` MCP tool for manual role discovery
  - Roles now emerge organically from actual usage patterns

### 🚨 Critical Path for MVP

The README "Simple Example" now works! The system can:
1. **User says**: "Fix the bug in the user authentication"
2. **System**: Captures query → Tracks AI's interpretation → Logs actions → Detects patterns → Applies learning

**✅ ACHIEVED**: MCP tools now use the new Interaction model to track full conversation cycles.

## Recent Achievements (Last Session)

### Hybrid Storage Implementation
- [x] Created VectorStore class for embedding storage
- [x] Implemented HybridStore combining SQLite and vector storage
- [x] Added embedding generation with sentence-transformers (optional)
- [x] Implemented semantic similarity search for queries
- [x] Added clustering and pattern discovery methods
- [x] Created embedding utilities for text preprocessing
- [x] Updated SQLite schema with complete_interactions table

### Interaction Model Implementation
- [x] Created complete Interaction dataclass with full lifecycle tracking
- [x] Implemented Action dataclass with status tracking
- [x] Added FeedbackType enum for categorizing user responses
- [x] Created InteractionLogger for tracking full conversation cycles
- [x] Implemented InteractionPatternDetector with NLP capabilities
- [x] Added methods for intent, success, and correction pattern detection

### Command Role System
- [x] Designed and implemented CommandRole data structures
- [x] Created RoleDiscoverer for pattern mining and role extraction
- [x] Implemented RoleManager for activation and management
- [x] Integrated roles with Sleep observation pipeline
- [x] Added role-based prompt enhancement
- [x] Created role resource endpoints and tools
- [x] Made sklearn optional for broader compatibility

### Documentation Improvements
- [x] Created comprehensive ARCHITECTURE.md
- [x] Moved DESIGN.md and TODO.md to docs/ directory
- [x] Updated CLAUDE.md to emphasize documentation-first approach
- [x] Renamed KBM to "Sleep" and central mind to "Wake"

## Architectural Refactoring Needed

### Complete Interaction Model
The current system only tracks individual actions, missing the complete user interaction flow. We need to refactor to capture:
- **User Query**: The actual natural language request
- **AI Interpretation**: What the AI understood
- **Action Sequence**: Uninterrupted sequence of actions taken
- **User Feedback**: Response to the complete action sequence

This will enable:
1. Learning user language patterns and preferences
2. Intent-to-action mapping
3. Success/failure pattern detection
4. Personalized language understanding
5. Better correction learning

## MVP Task List

### Phase 1: Core System Improvements

#### 1.1 Fix Pattern Detection Algorithm
**Priority**: High
- [x] Fix confidence score calculation in `PatternDetector`
- [x] Fix TypeError with None file paths in workflow detection
- [x] Implement proper sequence matching with fuzzy logic (Levenshtein distance)
- [x] Add minimum occurrence threshold (default: 3)
- [x] Add time-based weighting for recent patterns (exponential decay)
- [x] Fix workflow pattern detection test (sorted interactions by timestamp)
- [x] Write comprehensive tests for edge cases

#### 1.1.1 Infrastructure Updates
**Priority**: High
- [x] Change .ai directory to .bicamrl
- [x] Add proper logging configuration with rotation
- [x] Update all references in code and docs
- [x] Create logging utility module

#### 1.2 Enhance Memory Manager
**Priority**: High
- [x] Implement memory consolidation (active → working → episodic → semantic)
- [x] Add automatic cleanup for old active memories (counting implemented)
- [x] Implement context scoring algorithm
- [x] Add memory search with relevance ranking
- [x] Optimize database queries with proper indexes

#### 1.3 Refactor to Complete Interaction Model
**Priority**: Critical

##### Core Data Model Changes
- [x] Define new Interaction dataclass with:
  - [x] user_query: str
  - [x] ai_interpretation: str
  - [x] planned_actions: List[str]
  - [x] actions_taken: List[Action]
  - [x] execution_time: float
  - [x] user_feedback: Optional[str]
  - [x] feedback_type: FeedbackType enum
  - [x] success: bool
- [x] Create Action dataclass for individual actions
- [x] Define FeedbackType enum (APPROVAL, CORRECTION, FOLLOWUP, NONE)

##### Storage Layer Updates
- [x] Design new interactions table schema (complete_interactions)
- [x] Update SQLiteStore with interaction CRUD operations
- [x] Create HybridStore combining SQLite and vector storage
- [x] Implement vector similarity search for queries
- [x] Add indexes for query and interpretation fields
- [ ] Create data migration script for existing data

##### Pattern Detection Refactoring
- [x] Create InteractionPatternDetector class
- [x] Implement detect_intent_patterns() for query→action mappings
- [x] Implement detect_success_patterns() for positive feedback patterns
- [x] Implement detect_correction_patterns() for corrections
- [x] Add language pattern detection for user queries (simple keyword extraction)

##### Logging System Overhaul
- [x] Create InteractionLogger to track full cycles
- [x] Implement start_interaction() method
- [x] Implement log_interpretation() method
- [x] Implement log_action() for each action in sequence
- [x] Implement complete_interaction() with feedback capture
- [ ] Update MCP tools to use new logging

##### Learning Improvements
- [ ] Intent mapping: Learn user vocabulary → action sequences
- [ ] Correction learning: Track misinterpretations
- [ ] Success tracking: Identify effective patterns
- [ ] Personalization: Build user-specific language models

### Critical: Making "A Simple Example" Work

**Priority**: CRITICAL - Required for README example to function

#### MCP Tool Integration with Interaction Model ✅ COMPLETE
- [x] Update `log_interaction` to use InteractionLogger instead of just logging actions (REMOVED OLD METHOD)
- [x] Add `start_interaction` tool to capture user's original query
- [x] Add `log_ai_interpretation` tool to record what AI understood
- [x] Add `complete_interaction` tool to finalize with feedback
- [x] Update `get_relevant_context` to search similar queries and suggest actions

#### Interaction Tracking Flow ✅ COMPLETE
- [x] Implement mechanism to capture user's original query in MCP context
- [x] Connect query → AI interpretation → actions → feedback cycle
- [x] Store complete interactions in hybrid store with embeddings
- [ ] Add automatic interaction completion on session end (Nice to have)

#### Pattern Application in Real-time
- [x] Hook up `get_relevant_context` to search similar queries
- [x] Implement query-to-action mapping suggestions
- [x] Add suggested_actions based on similar successful interactions
- [ ] Enable confidence-based pattern recommendations with thresholds

#### Feedback Loop Connection
- [x] Update `record_feedback` to associate with recent interaction
- [x] Implement implicit success detection (no complaints = success)
- [x] Add feedback type inference from user messages
- [ ] Connect feedback to pattern confidence updates

#### 1.4 Improve Feedback Processing
**Priority**: Medium
- [ ] Add sentiment analysis for feedback
- [ ] Implement preference conflict resolution
- [ ] Add preference categories (code style, tools, patterns)
- [ ] Create preference inheritance (global → project → file)

### Phase 2: Knowledge Base Maintainer Integration

#### 2.1 Complete Sleep Implementation
**Priority**: High
- [x] Wire Sleep into server lifecycle management
- [x] Implement observation queue and processing
- [x] Add Sleep resource endpoints (status, config, insights, templates)
- [x] Implement insight generation algorithms
- [x] Add Sleep configuration validation

#### 2.2 Prompt Optimization
**Priority**: Medium
- [x] Integrate prompt optimizer with Sleep
- [ ] Implement A/B testing framework for prompts
- [ ] Add prompt performance metrics
- [x] Create prompt template library
- [x] Add context-aware prompt selection

#### 2.3 Multi-LLM Coordination
**Priority**: Medium
- [ ] Test OpenAI provider implementation
- [ ] Test Anthropic provider implementation
- [ ] Implement provider fallback logic
- [ ] Add cost tracking and optimization
- [ ] Create provider selection strategy

### Phase 3: Testing & Documentation

#### 3.1 Comprehensive Testing
**Priority**: High
- [x] Add integration tests for full workflow
- [x] Add stress tests for memory system
- [x] Test pattern detection with real data
- [x] Test Sleep with mock LLM responses
- [x] Add performance benchmarks

#### 3.2 Documentation
**Priority**: Medium
- [x] Update README with current features
- [x] Create user guide for MCP integration (getting-started.md)
- [x] Document configuration options (configuration.md)
- [x] Add troubleshooting guide (troubleshooting.md)
- [x] Create API reference (api-reference.md)
- [x] Create comprehensive ARCHITECTURE.md
- [x] Reorganize docs to docs/ directory
- [x] Update CLAUDE.md with documentation-first approach

### Phase 4: Testing & Efficacy Measurement

#### 4.1 Effectiveness Metrics
**Priority**: High
- [ ] Add A/B testing framework for prompt enhancement
- [ ] Implement pattern usefulness tracking
- [ ] Add productivity impact measurements
- [ ] Create user satisfaction metrics
- [ ] Build effectiveness dashboard

#### 4.2 Long-term Testing
**Priority**: Medium
- [ ] Add multi-week usage simulation tests
- [ ] Test semantic memory retrieval effectiveness
- [ ] Measure pattern quality over time
- [ ] Test memory consolidation accuracy at scale

### Phase 5: Production Hardening

#### 5.1 Error Handling
**Priority**: High
- [ ] Add graceful degradation for Sleep failures
- [ ] Implement retry logic for database operations
- [ ] Add circuit breakers for external services
- [ ] Improve error messages and logging
- [ ] Add health check endpoints

#### 5.2 Performance Optimization
**Priority**: Medium
- [ ] Profile and optimize hot paths
- [ ] Add caching for frequently accessed data
- [ ] Implement connection pooling
- [ ] Optimize pattern matching algorithms
- [ ] Add metrics collection

#### 5.3 Security & Privacy
**Priority**: High
- [ ] Add input validation for all tools
- [ ] Implement data sanitization
- [ ] Add rate limiting
- [ ] Ensure no sensitive data in logs
- [ ] Add audit logging

## Future Phases (Post-MVP)

### Phase 6: Probabilistic Enhancement
- [ ] Research GenJAX integration approach
- [ ] Design probabilistic pattern models
- [ ] Implement Bayesian preference learning
- [ ] Add uncertainty quantification
- [ ] Create probabilistic transition models

### Phase 7: Model-Based Planning
- [ ] Design world model schema
- [ ] Implement basic state/action representations
- [ ] Create planning algorithms (MCTS, beam search)
- [ ] Implement plan dispatch protocol
- [ ] Add execution monitoring

### Phase 8: Full Meta-Cognition
- [ ] Add custom planner generation
- [ ] Create active learning system
- [ ] Implement multi-objective optimization
- [ ] Add explainability features

## Development Guidelines

### Testing Strategy
1. Unit test each component in isolation
2. Integration test tool interactions
3. End-to-end test with mock MCP client
4. Performance test with realistic workloads

### Code Quality
1. Type hints for all functions
2. Docstrings following Google style
3. Ruff formatting
4. Pyright type checking
5. No emojis in code/docs

### Git Workflow
1. Feature branches for each task
2. PR with tests before merge
3. Squash commits for clean history
4. Tag releases (v0.1.0 for MVP)

## Success Criteria for MVP

1. **Memory System**: Reliably stores and retrieves interactions
2. **Pattern Detection**: Identifies repeated workflows with >80% accuracy
3. **Feedback Integration**: Adapts behavior based on preferences
4. **Sleep (if enabled)**: Provides useful insights and optimizations
5. **Performance**: <100ms response time for most operations
6. **Reliability**: >99% uptime with graceful error handling
7. **Documentation**: Clear setup and usage instructions

## Immediate Next Steps

### High Priority
1. **Test Role System**: Write comprehensive tests for the role discovery, management, and activation features
2. **Document Role System**: Update documentation to explain the role concept and how to use it
3. **Production Testing**: Test the system with real usage patterns
4. **Performance Profiling**: Identify and optimize bottlenecks in the interaction tracking flow

### Medium Priority
1. **A/B Testing Framework**: Implement A/B testing for open-source models
2. **Cost Tracking**: Add token usage and cost tracking for different LLM providers
3. **Automatic Session Management**: Implement automatic interaction completion on session end
4. **Advanced Feedback Processing**: Add sentiment analysis and preference conflict resolution

### Nice to Have
1. **Data Migration**: Create scripts to migrate old interaction data to new format
2. **Dashboard**: Create a web dashboard to visualize patterns and insights
3. **Export/Import**: Add ability to export/import learned patterns and preferences

## Notes

- Priority: High = blocking other work, Medium = important but not blocking, Low = nice to have
- Sleep features are optional but included in MVP for completeness
- Focus on robustness over features for MVP
- The core interaction model is now complete and working!
