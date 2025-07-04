# Claude Code Tool Permissions

This document explains how to configure and use tool permissions with Claude Code in Bicamrl.

## Overview

When using Claude Code, the model can invoke various tools to interact with your system:
- **Read** - Read files
- **Write** - Create or overwrite files
- **Edit** / **MultiEdit** - Modify existing files
- **Bash** - Execute shell commands
- **WebFetch** - Fetch web content
- **Grep** / **Glob** - Search files

By default, Claude Code will prompt for permission before using these tools. Bicamrl provides a flexible permission system to control this behavior.

## Permission Modes

Claude Code supports three permission modes:

1. **`default`** - Prompts for each tool use (recommended)
2. **`acceptEdits`** - Automatically accepts file edits
3. **`bypassPermissions`** - Bypasses all permission checks (use with caution!)

## Basic Configuration

Add permission settings to your `Mind.toml`:

```toml
[llm_providers.claude_code]
type = "claude_code"
model = "claude-opus-4-20250514"
permission_mode = "default"  # or "acceptEdits" or "bypassPermissions"
```

## Custom Permission Control

For fine-grained control, you can use a custom MCP permission server:

### 1. Configure the Permission Server

```toml
[llm_providers.claude_code]
type = "claude_code"
model = "claude-opus-4-20250514"
permission_mode = "default"

# Specify the custom permission tool
permission_prompt_tool = "mcp__bicamrl-permissions__approval_prompt"

# List MCP tools that Claude can use
mcp_tools = [
    "mcp__bicamrl-permissions__approval_prompt",
    "mcp__bicamrl-permissions__update_permission_policy",
    "mcp__bicamrl-permissions__list_permission_policies"
]

# Configure the MCP server
[llm_providers.claude_code.mcp_servers.bicamrl-permissions]
type = "stdio"
command = "python"
args = ["-m", "bicamrl.tui.permission_server"]
```

### 2. Default Permission Policies

The permission server has default policies for common tools:

**Always Allowed** (safe, read-only):
- `Read`
- `Grep`
- `Glob`
- `WebFetch`

**Always Denied** (potentially dangerous):
- `Bash`

**Ask User** (modifies files):
- `Write`
- `Edit`
- `MultiEdit`

### 3. Updating Permission Policies

You can update policies at runtime by asking Claude:

```
User: Always allow Edit operations
Claude: [Updates permission policy - Edit will now be automatically allowed]

User: Deny all Bash commands
Claude: [Updates permission policy - Bash will now be automatically denied]

User: Show current permission policies
Claude: [Lists all permission policies]
```

## Security Best Practices

1. **Start with `default` mode** - This ensures you're aware of what tools Claude is using

2. **Be cautious with `Bash`** - Shell commands can be dangerous. Consider keeping this tool denied

3. **Review file operations** - Even with permissions, review what files Claude is modifying

4. **Use `acceptEdits` carefully** - Only use this mode when you trust the operations being performed

5. **Avoid `bypassPermissions`** - This mode should only be used in isolated environments

## Integration with TUI

The TUI can show permission requests in the status bar:

- When a tool requires permission, you'll see a prompt in the status bar
- Use keyboard shortcuts to approve/deny (coming soon)
- Permission decisions can be remembered for the session

## Example Use Cases

### Safe Exploration Mode
```toml
permission_mode = "default"
# User approves each action
```

### Development Mode
```toml
permission_mode = "acceptEdits"
# Auto-accepts file edits but still prompts for Bash
```

### Isolated Testing
```toml
permission_mode = "bypassPermissions"
# Use only in Docker/VM environments
```

## Troubleshooting

### Permission Tool Not Found
If you see errors about the permission tool not being found, ensure:
1. The MCP server name matches in the tool name and server config
2. The Python module path is correct
3. PYTHONPATH is set correctly in the server environment

### Tools Not Working
If tools aren't being invoked:
1. Check that tools are listed in `allowed_tools`
2. Verify the permission mode is set correctly
3. Check logs for permission denial messages

### Custom Permission Logic
To implement custom permission logic:
1. Subclass `PermissionManager`
2. Override the `check_permission` method
3. Implement your custom rules
