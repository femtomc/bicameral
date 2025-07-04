# Sleep User Guide

## Overview

The Sleep is a meta-cognitive background processing system that observes, learns from, and optimizes AI interactions. It acts as a "supervisor" that helps the main AI instance work more effectively over time.

## Key Concepts

### Two-Tier Intelligence System

1. **Wake Layer** (Main AI Instance, e.g., Claude)
   - Focuses on immediate tasks
   - Works with current context
   - Generates code and answers questions

2. **Sleep** (Background Processing)
   - Observes interaction patterns
   - Maintains and curates knowledge
   - Generates optimized prompts
   - Identifies knowledge gaps
   - Suggests context improvements

## Configuration

### Quick Start

1. Set environment variables:
```bash
export SLEEP_LAYER_ENABLED=true
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

2. Or create a configuration file:
```bash
cp bicamrl_config.example.json .bicamrl/config.json
# Edit .bicamrl/config.json with your API keys
```

### Configuration Options

```json
{
  "sleep": {
    "enabled": true,                    // Enable/disable Sleep
    "batch_size": 10,                   // Interactions to process at once
    "analysis_interval": 300,           // Seconds between deep analysis
    "min_confidence": 0.7,              // Minimum confidence for insights
    "llm_providers": {
      // Configure different LLMs for different roles
      "roles": {
        "analyzer": "openai",           // For pattern analysis
        "generator": "claude",          // For content generation
        "enhancer": "claude",           // For prompt enhancement
        "optimizer": "openai"           // For optimization tasks
      }
    }
  }
}
```

## Using Sleep with Claude Code

### Automatic Observation

When Sleep is enabled, it automatically observes all interactions and learns patterns. No manual intervention needed!

### Manual Observation

You can explicitly report interactions:

```
Use tool: observe_interaction
{
  "interaction_type": "code_generation",
  "query": "implement user authentication",
  "response": "Created auth.py with JWT implementation",
  "tokens_used": 1500,
  "latency": 2.3,
  "success": true
}
```

### Getting Prompt Recommendations

Before sending a complex query:

```
Use tool: optimize_prompt
{
  "prompt": "fix the bug in authentication",
  "task_type": "bug_fix"
}
```

Response:
```json
{
  "original": "fix the bug in authentication",
  "optimized": "Debug and fix the following issue:\n\nError: Authentication failing\nFile: auth.py\n\nProject patterns:\n- Use JWT tokens\n- bcrypt for passwords\n\nProvide:\n1. Root cause analysis\n2. Fix implementation\n3. Prevention strategy",
  "context_additions": [
    {"type": "file", "path": "auth.py", "reason": "mentioned in prompt"},
    {"type": "file", "path": "tests/test_auth.py", "reason": "frequently accessed"}
  ],
  "confidence": 0.85,
  "reasoning": "Detected intent: bug_fix. Found 3 relevant patterns. Applied 'bug_fix' template"
}
```

### Accessing Sleep Resources

View insights and recommendations:

```
@bicamrl/sleep_layer/insights       # View Sleep insights
@bicamrl/sleep_layer/prompt-templates  # View optimized templates
```

## Benefits

### 1. Automatic Context Enhancement
Sleep automatically suggests relevant files and patterns based on your query.

### 2. Learning from Mistakes
When interactions fail or take too long, Sleep analyzes the root cause and prevents similar issues.

### 3. Prompt Optimization
Sleep learns which prompt structures work best for your project and automatically enhances queries.

### 4. Pattern Recognition
Discovers workflows and conventions specific to your codebase.

## Example Workflow

### Without Sleep:
```
You: "implement password reset"
Claude: [Generic implementation]
You: "no, use our email service and follow our patterns"
Claude: [Better implementation]
You: "also need to use our token format"
Claude: [Final implementation]
```

### With Sleep:
```
You: "implement password reset"

[Sleep automatically enhances the prompt with project patterns, conventions, and relevant files]

Claude: [Correct implementation following all project patterns on first try]
```

## Advanced Features

### Multi-LLM Coordination

Sleep can use different LLMs for different tasks:
- GPT-4 for analysis and pattern recognition
- Claude for code generation and enhancement
- Local models for privacy-sensitive operations

### Consensus Analysis

For critical decisions, Sleep can get consensus from multiple LLMs:

```
Use tool: get_sleep_layer_recommendation
{
  "query": "best approach for migrating database schema",
  "context": {"current_schema": "v1", "target_schema": "v2"}
}
```

### Performance Monitoring

Sleep tracks:
- Response times
- Token usage
- Success rates
- Pattern effectiveness

## Best Practices

### 1. Let Sleep Learn
Give it time to observe patterns. The more interactions, the better it gets.

### 2. Provide Feedback
When Sleep suggestions don't work:
```
/feedback correct "Sleep suggested wrong pattern for async code"
```

### 3. Review Insights
Periodically check Sleep insights:
```
/memory show patterns
@bicamrl/sleep_layer/insights
```

### 4. Tune Confidence Thresholds
Adjust `min_confidence` based on your needs:
- Higher (0.8+): Only very confident suggestions
- Lower (0.6+): More suggestions, some may be wrong

## Troubleshooting

### Sleep Not Working?

1. Check configuration:
```bash
cat .bicamrl/config.json | grep -A5 "sleep"
```

2. Verify API keys:
```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

3. Check logs:
```bash
# Look for Sleep-related messages
grep "Sleep" logs/bicamrl.log
```

### High Latency?

- Reduce `batch_size` for faster processing
- Increase `analysis_interval` to reduce frequency
- Use faster LLM models in configuration

### Too Many/Few Suggestions?

Adjust `min_confidence`:
- Too many suggestions: Increase to 0.8 or 0.9
- Too few suggestions: Decrease to 0.6 or 0.5

## Future Enhancements

1. **Predictive Context Loading**: Sleep will predict needed context before you ask
2. **Cross-Project Learning**: Share patterns across projects
3. **Team Knowledge Sharing**: Merge insights from multiple developers
4. **Visual Analytics**: Dashboard showing Sleep insights and performance

## Summary

The Sleep transforms your AI assistant from a stateless tool into an intelligent collaborator that:
- Learns your project's patterns
- Optimizes interactions automatically
- Prevents repeated mistakes
- Gets better over time

Enable Sleep today and watch your AI assistant become a true expert in your codebase!
