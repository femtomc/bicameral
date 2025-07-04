# Developer Feedback Guide

## How to Train Your AI Assistant

This guide explains how to provide effective feedback to help Claude Code learn your project's patterns and preferences.

## Quick Feedback Commands

### 1. Corrections - When the AI makes a mistake
```bash
# Correct a wrong assumption
ai-feedback correct "We don't use callbacks in this project - everything is async/await"

# Correct a wrong pattern
ai-feedback correct "API routes should be in /routes not /controllers"

# Correct style issues
ai-feedback correct "Use 2 spaces for indentation, not 4"
```

### 2. Preferences - Teaching your style
```bash
# Code style preferences
ai-feedback prefer "Use named exports instead of default exports"
ai-feedback prefer "Put interfaces in separate .types.ts files"
ai-feedback prefer "Use early returns for validation"

# Workflow preferences
ai-feedback prefer "Write tests before implementation"
ai-feedback prefer "Update documentation in the same commit"
ai-feedback prefer "Use conventional commits for messages"
```

### 3. Patterns - Teaching workflows
```bash
# Common workflows
ai-feedback pattern "When adding API endpoints: update route → add controller → add service → write tests"
ai-feedback pattern "Always run lint before committing"
ai-feedback pattern "Database changes need a migration file"

# Debugging patterns
ai-feedback pattern "Timeout errors in tests usually mean increase Jest timeout"
ai-feedback pattern "CORS errors: check the whitelist in config/cors.js"
```

## Inline Feedback in Code

Add special comments that the AI will learn from:

```typescript
// @ai-pattern: This service always returns DTOs, never raw entities
export class UserService {
  async findUser(id: string): Promise<UserDto> {
    // @ai-feedback: Always check cache first
    const cached = await this.cache.get(`user:${id}`);
    if (cached) return cached;

    // @ai-preference: Use findOneOrFail for better error handling
    const user = await this.repo.findOneOrFail(id);

    // @ai-pattern: Always transform to DTO before returning
    return UserDto.fromEntity(user);
  }
}
```

```python
# @ai-context: This module handles all authentication logic
class AuthManager:
    # @ai-pattern: All auth methods should be async
    async def login(self, credentials):
        # @ai-critical: NEVER log passwords
        self.logger.info(f"Login attempt for {credentials.username}")

        # @ai-preference: Use constant-time comparison for passwords
        if not await self.verify_password(credentials):
            # @ai-pattern: Always rate-limit failed attempts
            await self.rate_limiter.record_failure(credentials.username)
            raise AuthenticationError()
```

## Feedback During Sessions

### Real-time corrections
When Claude suggests something, you can immediately correct:

```
Claude: "I'll add this to the controllers folder..."
You: ai-feedback correct "We put controllers in src/api/endpoints/"
```

### Confirming good patterns
When Claude does something right, reinforce it:

```
Claude: "I'll write the test first before implementing..."
You: ai-feedback confirm "Yes, always TDD for new features"
```

## Advanced Feedback Patterns

### 1. Contextual Rules
```bash
# Rules that apply in specific contexts
ai-feedback rule "In test files: use 'describe' and 'it', not 'test'"
ai-feedback rule "In API routes: always validate input with Joi"
ai-feedback rule "In services: never access req/res objects directly"
```

### 2. Anti-patterns to Avoid
```bash
# Things the AI should never do
ai-feedback avoid "Don't use 'any' type in TypeScript"
ai-feedback avoid "Don't commit console.log statements"
ai-feedback avoid "Don't modify generated files (*.generated.ts)"
```

### 3. Project-Specific Knowledge
```bash
# Domain knowledge
ai-feedback context "User IDs are UUIDs, not integers"
ai-feedback context "All monetary values are stored in cents"
ai-feedback context "Dates are always UTC in the database"
```

## Viewing What AI Has Learned

### Check current knowledge
```bash
# See all learned patterns
ai-memory show patterns

# See preferences
ai-memory show preferences

# See recent feedback
ai-memory show feedback --recent

# Search for specific learnings
ai-memory search "authentication"
```

### Export and Share Knowledge
```bash
# Export for team member
ai-memory export --format json > project-ai-knowledge.json

# Import from team member
ai-memory import project-ai-knowledge.json

# Merge team knowledge
ai-memory merge --from teammate-export.json
```

## Best Practices

### 1. Be Specific
```bash
# ❌ Too vague
ai-feedback prefer "write better code"

# ✅ Specific and actionable
ai-feedback prefer "use descriptive variable names (full words, not abbreviations)"
```

### 2. Provide Context
```bash
# ❌ Missing context
ai-feedback correct "don't do that"

# ✅ Clear context
ai-feedback correct "don't use findOne() - use findOneOrFail() for better error handling"
```

### 3. Teach Incrementally
Start with the most important patterns and gradually add more nuanced preferences:

```bash
# Week 1: Core patterns
ai-feedback pattern "All API routes need authentication middleware"

# Week 2: Style preferences
ai-feedback prefer "Use object destructuring in function parameters"

# Week 3: Advanced patterns
ai-feedback pattern "Complex queries should use the query builder, not raw SQL"
```

## Feedback Templates

### For New Projects
```bash
# Essential setup feedback
ai-feedback context "This is a TypeScript project with strict mode"
ai-feedback pattern "File structure: src/{feature}/{layer}/"
ai-feedback prefer "Use functional components with hooks (no class components)"
ai-feedback rule "All async functions need error handling"
```

### For Legacy Projects
```bash
# Help AI understand legacy patterns
ai-feedback context "Legacy code uses callbacks - new code should use async/await"
ai-feedback pattern "Old modules are in /lib, new ones in /src"
ai-feedback avoid "Don't refactor legacy code without explicit request"
```

## Troubleshooting

### If AI isn't learning your patterns
1. Check feedback was recorded: `ai-memory show feedback --recent`
2. Ensure hooks are running: Check `.claude/hooks/` permissions
3. Look for conflicting feedback: `ai-memory conflicts`
4. Reset specific patterns: `ai-memory forget pattern <pattern-id>`

### If AI is too aggressive with patterns
```bash
# Reduce confidence in specific patterns
ai-feedback adjust pattern-id --confidence 0.5

# Disable pattern temporarily
ai-feedback disable pattern-id
```

## Integration with Team Workflows

### Code Reviews
```bash
# After code review feedback
ai-feedback review "PR #123: Always handle the empty array case"
ai-feedback review "PR #124: Good pattern for error handling - use everywhere"
```

### Post-Incident Learning
```bash
# After fixing a bug
ai-feedback incident "Bug #456: Always validate email format before sending"
ai-feedback incident "Incident #12: Check rate limits before bulk operations"
```

## Summary

Effective feedback helps Claude Code become a better collaborator over time. Remember:

1. **Be specific** - Clear, actionable feedback works best
2. **Be consistent** - Reinforce patterns you want to stick
3. **Be patient** - Learning happens over multiple interactions
4. **Be collaborative** - Share learnings with your team

The more feedback you provide, the more effective your AI assistant becomes at understanding and working with your specific codebase.
