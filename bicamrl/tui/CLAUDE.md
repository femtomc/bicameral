# TUI Module Instructions

**IMPORTANT**: This module bridges Python and Rust for the terminal UI. Handle async operations carefully.

## Module Purpose

The TUI module provides:
- **Rust Integration**: PyO3 bindings to Rust TUI
- **Async Bridge**: Python asyncio ↔ Rust tokio
- **Streaming UI**: Real-time updates from Claude Code
- **Permission System**: Tool approval workflows
- **Wake Agent**: Interactive AI conversations

## Critical Architecture Decisions

### Async Architecture
**YOU MUST**:
- Use AsyncBridge for Python→Rust callbacks
- Handle streaming responses incrementally
- Never block the UI thread
- Properly cleanup async tasks

### Permission Flow
```
Claude wants tool → Permission request → TUI popup → User decision → Callback
```

**CRITICAL**: Permission server runs on separate port (8765)

### Message Flow
1. User types in Rust TUI
2. Rust calls Python callback
3. Python processes with Wake agent
4. Streaming response back to Rust
5. Rust renders incrementally

## Key Files

- `rust_tui.py` - Main Python→Rust bridge
- `async_bridge.py` - Asyncio↔Tokio converter
- `wake_agent.py` - Claude Code integration
- `streaming_handler.py` - Handles streaming responses
- `permission_server.py` - MCP permission tool
- `permission_http_server.py` - HTTP server for permissions
- `chat_widget.py` - Python-native chat (backup)

## Common Operations

### Starting the TUI
```python
tui = BicamrlRustTUI(
    db_path=".bicamrl/memory",
    config=config,
    message_callback=callback
)
await tui.run()
```

### Handling streaming
```python
handler = StreamingHandler(callback)
async for chunk in claude_response:
    await handler.handle_chunk(chunk)
```

### Permission handling
```python
manager = PermissionManager()
manager.add_policy("read_file", "always_allow")
decision = await manager.request_permission(tool, args)
```

## Rust FFI Patterns

**IMPORTANT**: Follow these patterns:

### Python→Rust
```python
self._tui.add_message(role, content)
self._tui.update_spinner(thinking=True)
```

### Rust→Python
```rust
Python::with_gil(|py| {
    callback.call1(py, (message,))
})
```

### Thread Safety
- Use Arc<Mutex<>> in Rust
- Use threading.Lock in Python
- Convert between them carefully

## Testing

Run TUI tests:
```bash
pixi run python -m pytest tests/test_rust_tui.py -v
pixi run python -m pytest tests/test_streaming_handler.py -v
pixi run python -m pytest tests/test_permission_flow.py -v

# Manual testing
pixi run tui
```

## Common Pitfalls

- **GIL deadlocks**: Release GIL before long operations
- **Async leaks**: Always cleanup tasks
- **Permission loops**: Check server is running
- **Memory leaks**: Rust objects need explicit cleanup
- **Unicode issues**: Test with emojis/special chars

## Debugging Tips

1. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. Check Rust panics:
```bash
RUST_BACKTRACE=1 pixi run tui
```

3. Monitor permission server:
```bash
curl http://localhost:8765/health
```

## Integration Points

- **Wake agent**: Processes user messages
- **Memory core**: Stores interactions
- **Permission system**: MCP tool integration
- **Claude Code SDK**: Streaming responses
