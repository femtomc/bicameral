# Troubleshooting Guide

This guide helps resolve common issues with Bicamrl.

## Installation Issues

### "bicamrl: command not found"

**Cause**: Bicamrl not in PATH or not installed correctly.

**Solution**:
```bash
# Verify installation
pip show bicamrl

# If not installed
pip install bicamrl

# If installed but not in PATH, use full Python module
python -m bicamrl.server
```

### Import errors

**Cause**: Missing dependencies or Python version mismatch.

**Solution**:
```bash
# Check Python version (need 3.9+)
python --version

# Reinstall with all dependencies
pip install --upgrade bicamrl[all]
```

## MCP Connection Issues

### Claude Desktop doesn't show Bicamrl

**Cause**: Configuration file syntax error or wrong location.

**Solution**:
1. Verify config file location:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. Validate JSON syntax:
```bash
python -m json.tool < claude_desktop_config.json
```

3. Check server configuration:
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

### "Failed to start MCP server"

**Cause**: Python path incorrect or permissions issue.

**Solution**:
1. Find your Python path:
```bash
which python
# or
where python  # Windows
```

2. Update config with full path:
```json
{
  "command": "/usr/bin/python3",
  "args": ["-m", "bicamrl.server"]
}
```

## Memory Issues

### Patterns not being detected

**Cause**: Not enough repetitions or time gaps too large.

**Solution**:
- Patterns need 3+ repetitions by default
- Actions must be within 5 minutes to group as workflow
- Check pattern detection settings:
```json
{
  "pattern_detection": {
    "min_frequency": 2,  // Lower threshold
    "workflow_gap_minutes": 10  // Increase gap tolerance
  }
}
```

### Memory not persisting between sessions

**Cause**: Database location changing or permissions issue.

**Solution**:
1. Check database location:
```bash
ls -la ~/.bicamrl/memory/
```

2. Verify write permissions:
```bash
touch ~/.bicamrl/memory/test.txt
rm ~/.bicamrl/memory/test.txt
```

3. Set fixed path in config:
```json
{
  "env": {
    "MEMORY_DB_PATH": "/absolute/path/to/.bicamrl/memory"
  }
}
```

### Database errors

**Cause**: Corrupted database or version mismatch.

**Solution**:
1. Backup existing data:
```bash
cp -r ~/.bicamrl ~/.bicamrl.backup
```

2. Reset database:
```bash
rm ~/.bicamrl/memory/memory.db
```

3. Bicamrl will create fresh database on next run

## Performance Issues

### Slow response times

**Cause**: Large database or inefficient queries.

**Solution**:
1. Check database size:
```bash
du -h ~/.bicamrl/memory/memory.db
```

2. Run memory consolidation:
```
Ask Claude: "Run memory consolidation"
```

3. Adjust performance settings:
```json
{
  "performance": {
    "max_interactions": 50000,
    "pattern_check_interval": 600
  }
}
```

### High memory usage

**Cause**: Too many patterns or large context.

**Solution**:
- Increase consolidation frequency
- Limit pattern detection scope
- Clear old patterns:
```
Ask Claude: "Clear patterns older than 30 days"
```

## Sleep Layer Issues

### "Sleep Layer not enabled"

**Cause**: Missing configuration or API keys.

**Solution**:
1. Create `bicamrl_config.json`:
```json
{
  "kbm": {
    "enabled": true,
    "llm_providers": {
      "mock": {}  // For testing
    }
  }
}
```

2. With real providers:
```bash
export OPENAI_API_KEY=sk-...
```

### API errors

**Cause**: Invalid API key or rate limits.

**Solution**:
- Verify API key is valid
- Check API usage/limits on provider dashboard
- Add retry configuration:
```json
{
  "kbm": {
    "llm_providers": {
      "openai": {
        "max_retries": 5,
        "timeout": 60
      }
    }
  }
}
```

## Debug Mode

Enable detailed logging to diagnose issues:

```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

Or via environment:
```bash
export BICAMERAL_LOG_LEVEL=DEBUG
```

Check logs at:
- `~/.bicamrl/logs/bicamrl.log`
- `~/.bicamrl/logs/error.log`

## Getting Help

If these solutions don't resolve your issue:

1. **Check logs** for specific error messages
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Error messages
   - Configuration (without API keys)
   - Steps to reproduce
   - System information (OS, Python version)

## Common Error Messages

### "ValueError: Invalid configuration"
Check JSON syntax and required fields in config files.

### "PermissionError: [Errno 13]"
File permissions issue. Check write access to `.bicamrl` directory.

### "sqlite3.OperationalError: database is locked"
Another process is using the database. Restart Claude Desktop.

### "asyncio.TimeoutError"
Network or API timeout. Check internet connection and API status.

### "ImportError: No module named 'mcp'"
MCP not installed. Run: `pip install mcp`
