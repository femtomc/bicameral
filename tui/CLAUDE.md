# Rust TUI Module Instructions

**IMPORTANT**: This is the Rust implementation of Bicamrl's terminal UI. It integrates with Python via PyO3.

## Module Purpose

The Rust TUI provides:
- **High-Performance Rendering**: Using ratatui/crossterm
- **Real-time Updates**: Smooth streaming of AI responses
- **Unicode Support**: Proper emoji and special character handling
- **Theme System**: Andromeda-inspired color scheme
- **Python Integration**: PyO3 bindings for Python callbacks

## Critical Architecture Decisions

### Threading Model
**YOU MUST**:
- Use Arc<Mutex<>> for shared state
- Run UI in separate thread from Python
- Use channels for thread communication
- Handle Python GIL properly

### Widget Architecture
```rust
App State → UI Renderer → Terminal
     ↑           ↓
  Python ← → Event Loop
```

### Message Flow
1. User input in terminal
2. Event loop processes key
3. Callback to Python (with GIL)
4. Python returns response
5. Update app state
6. Render new frame

## Key Files

- `lib.rs` - PyO3 module definition and main TUI struct
- `app.rs` - Application state management
- `ui.rs` - Main rendering logic
- `theme.rs` - Andromeda theme colors
- `markdown.rs` - Markdown rendering
- `code_highlighter.rs` - Syntax highlighting
- `spinner.rs` - Thinking animation
- `unicode_utils.rs` - Unicode width handling

## Common Operations

### Python Integration
```rust
#[pyclass]
pub struct BicamrlTUI {
    app: Arc<Mutex<App>>,
    // ...
}

#[pymethods]
impl BicamrlTUI {
    #[new]
    fn new(callback: PyObject) -> Self { ... }

    fn add_message(&self, role: &str, content: &str) { ... }
}
```

### Rendering
```rust
terminal.draw(|f| {
    ui::draw(f, &app);
})?;
```

### Event Handling
```rust
if event::poll(Duration::from_millis(50))? {
    if let Event::Key(key) = event::read()? {
        match key.code {
            KeyCode::Enter => { /* send message */ }
            KeyCode::Esc => { /* cancel/exit */ }
            // ...
        }
    }
}
```

## Building

```bash
# Development build
pixi run build-dev

# Release build
pixi run build

# Run tests
cargo test
```

## Unicode Handling

**CRITICAL**: Use unicode-width crate:
```rust
use unicode_width::UnicodeWidthStr;

let width = text.width(); // NOT text.len()
```

## Theme System

Colors are defined in `theme.rs`:
```rust
pub const BACKGROUND: Color = Color::Rgb(19, 22, 30);
pub const FOREGROUND: Color = Color::Rgb(196, 197, 207);
pub const PURPLE: Color = Color::Rgb(142, 94, 221);
// ...
```

## Testing

Run Rust tests:
```bash
cargo test
cargo clippy -- -W warnings
cargo fmt -- --check
```

Python integration tests:
```bash
pixi run python -m pytest tests/test_rust_tui.py -v
```

## Common Pitfalls

- **String lifetimes**: Use String, not &str for PyO3
- **Panic handling**: Catch panics at FFI boundary
- **Terminal cleanup**: Always restore on exit
- **Unicode width**: Text length ≠ display width
- **Color support**: Check terminal capabilities

## Performance Tips

- Minimize redraws (only on state change)
- Use StatefulWidget for complex components
- Batch Python callbacks when possible
- Profile with `cargo flamegraph`

## Debugging

1. Enable Rust logging:
```bash
RUST_LOG=debug pixi run tui
```

2. Use dbg! macro:
```rust
dbg!(&app.messages);
```

3. Test without Python:
```bash
cargo run --example standalone_ui
```

## Integration Points

- **Python TUI module**: Via PyO3 bindings
- **Terminal**: Via crossterm
- **Event loop**: Tokio for async
- **Theme**: Shared with Python side
