use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use pyo3::prelude::*;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{
    io,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::mpsc;

mod app;
mod code_highlighter;
mod markdown;
mod spinner;
mod theme;
mod ui;
mod unicode_utils;
mod widgets;

use app::{App, PopupType};

#[pyclass]
pub struct BicamrlTUI {
    app: Arc<Mutex<App>>,
    tx: Option<mpsc::Sender<AppEvent>>,
    handle: Option<std::thread::JoinHandle<()>>,
    callback: Option<PyObject>,
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Quit,
    Refresh,
    KeyPress(KeyCode),
    UpdateStats(Stats),
    AddMessage(String, MessageType),
    AddStreamingText(String),
    AddSystemMessage(String),
    SendChatMessage(String),
    UpdateInput(String),
    StartThinking,
    StopThinking,
    UpdateSpinner,
    UpdateTokenCount(usize),
    ShowToolPermission(String, String),
    ShowToolUse(String),
    ClosePopup,
}

#[derive(Debug, Clone)]
pub enum MessageType {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct Stats {
    #[pyo3(get, set)]
    pub total_interactions: usize,
    #[pyo3(get, set)]
    pub total_patterns: usize,
    #[pyo3(get, set)]
    pub success_rate: f64,
    #[pyo3(get, set)]
    pub active_memories: usize,
    #[pyo3(get, set)]
    pub tokens_used: usize,
    #[pyo3(get, set)]
    pub active_sessions: usize,
}

#[pymethods]
impl Stats {
    #[new]
    fn new() -> Self {
        Self {
            total_interactions: 0,
            total_patterns: 0,
            success_rate: 0.0,
            active_memories: 0,
            tokens_used: 0,
            active_sessions: 0,
        }
    }
}

#[pymethods]
impl BicamrlTUI {
    #[new]
    fn new() -> PyResult<Self> {
        let app = Arc::new(Mutex::new(App::new()));

        Ok(Self {
            app,
            tx: None,
            handle: None,
            callback: None,
        })
    }

    fn set_callback(&mut self, callback: PyObject) {
        self.callback = Some(callback);
    }

    fn run(&mut self, py: Python) -> PyResult<()> {
        let (tx, mut rx) = mpsc::channel(1000); // Increase buffer size for streaming
        self.tx = Some(tx.clone());

        let app = self.app.clone();
        let callback = self.callback.clone();

        // We need to handle the callback in a way that's safe across threads
        let (callback_tx, callback_rx) = std::sync::mpsc::channel::<String>();

        // Start a thread to handle Python callbacks
        let tx_clone = tx.clone();
        if let Some(cb) = callback {
            std::thread::spawn(move || {
                while let Ok(msg) = callback_rx.recv() {
                    // Acquire GIL only when needed
                    Python::with_gil(|py| {
                        // Call Python and get response
                        match cb.call1(py, (msg.clone(),)) {
                            Ok(response) => {
                                if let Ok(response_str) = response.extract::<String>(py) {
                                    // Handle different response types
                                    if response_str.starts_with("__STREAMING__:") {
                                        // Parse streaming update
                                        if let Some(json_str) =
                                            response_str.strip_prefix("__STREAMING__:")
                                        {
                                            // Use serde_json for proper JSON parsing with Unicode support
                                            if let Ok(update) =
                                                serde_json::from_str::<serde_json::Value>(json_str)
                                            {
                                                let update_type = update
                                                    .get("type")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("");
                                                let content = update
                                                    .get("content")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("")
                                                    .to_string();
                                                let is_complete = update
                                                    .get("is_complete")
                                                    .and_then(|v| v.as_bool())
                                                    .unwrap_or(false);

                                                // Handle based on type
                                                match update_type {
                                                    "text" => {
                                                        if is_complete && !content.is_empty() {
                                                            // This is a complete message from non-streaming provider
                                                            let _ = tx_clone.blocking_send(
                                                                AppEvent::AddMessage(
                                                                    content,
                                                                    MessageType::Assistant,
                                                                ),
                                                            );
                                                        } else if !content.is_empty() {
                                                            // This is streaming text
                                                            let _ = tx_clone.blocking_send(
                                                                AppEvent::AddStreamingText(content),
                                                            );
                                                        }
                                                    }
                                                    "system" => {
                                                        // Check for special messages
                                                        if content.starts_with("__UPDATE_TOKENS__:")
                                                        {
                                                            if let Some(token_str) = content
                                                                .strip_prefix("__UPDATE_TOKENS__:")
                                                            {
                                                                if let Ok(tokens) =
                                                                    token_str.parse::<usize>()
                                                                {
                                                                    let _ = tx_clone.blocking_send(
                                                                        AppEvent::UpdateTokenCount(
                                                                            tokens,
                                                                        ),
                                                                    );
                                                                }
                                                            }
                                                        } else if content.starts_with(
                                                            "__TOOL_PERMISSION_REQUEST__:",
                                                        ) {
                                                            // Parse tool permission request
                                                            if let Some(perm_str) = content
                                                                .strip_prefix(
                                                                    "__TOOL_PERMISSION_REQUEST__:",
                                                                )
                                                            {
                                                                let parts: Vec<&str> = perm_str
                                                                    .splitn(2, ':')
                                                                    .collect();
                                                                if parts.len() >= 1 {
                                                                    let tool_name =
                                                                        parts[0].to_string();
                                                                    let tool_input = if parts.len()
                                                                        > 1
                                                                    {
                                                                        // Try to pretty-print JSON
                                                                        if let Ok(parsed) =
                                                                            serde_json::from_str::<
                                                                                serde_json::Value,
                                                                            >(
                                                                                parts[1]
                                                                            )
                                                                        {
                                                                            serde_json::to_string_pretty(&parsed).unwrap_or_else(|_| parts[1].to_string())
                                                                        } else {
                                                                            parts[1].to_string()
                                                                        }
                                                                    } else {
                                                                        String::new()
                                                                    };
                                                                    let _ = tx_clone.blocking_send(AppEvent::ShowToolPermission(tool_name, tool_input));
                                                                }
                                                            }
                                                        } else if content == "__CLOSE_POPUP__" {
                                                            // Close any open popup (e.g., tool running popup)
                                                            let _ = tx_clone.blocking_send(
                                                                AppEvent::ClosePopup,
                                                            );
                                                        } else if !content.is_empty() {
                                                            // Show other system messages as they contain important info like usage limits
                                                            let _ = tx_clone.blocking_send(
                                                                AppEvent::AddSystemMessage(content),
                                                            );
                                                        }
                                                    }
                                                    "usage" => {
                                                        // Show usage information
                                                        if !content.is_empty() {
                                                            let _ = tx_clone.blocking_send(
                                                                AppEvent::AddSystemMessage(content),
                                                            );
                                                        }
                                                    }
                                                    "tool_use" => {
                                                        // Extract tool name from content
                                                        if content.starts_with("Using tool: ") {
                                                            if let Some(tool_name) =
                                                                content.strip_prefix("Using tool: ")
                                                            {
                                                                let _ = tx_clone.blocking_send(
                                                                    AppEvent::ShowToolUse(
                                                                        tool_name.to_string(),
                                                                    ),
                                                                );
                                                            }
                                                        }
                                                    }
                                                    "tool_result" => {
                                                        // Close the tool use popup when done
                                                        let _ = tx_clone
                                                            .blocking_send(AppEvent::ClosePopup);
                                                    }
                                                    _ => {
                                                        // Default to streaming text
                                                        if !content.is_empty() {
                                                            let _ = tx_clone.blocking_send(
                                                                AppEvent::AddStreamingText(content),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else if response_str == "__NO_STREAMING__" {
                                        // No streaming updates available
                                    } else if response_str == "__PROCESSING__" {
                                        // Message is being processed asynchronously
                                        // Response will come through streaming
                                    } else if response_str == "__INTERRUPTED__" {
                                        // Show interruption as a regular message
                                        let _ = tx_clone.blocking_send(AppEvent::AddMessage(
                                            "Processing interrupted.".to_string(),
                                            MessageType::Assistant,
                                        ));
                                    } else if response_str == "__NO_ACTIVE_TASK__" {
                                        // No message needed - nothing was interrupted
                                    } else if response_str == "__START_THINKING__" {
                                        let _ = tx_clone.blocking_send(AppEvent::StartThinking);
                                    } else if response_str == "__STOP_THINKING__" {
                                        let _ = tx_clone.blocking_send(AppEvent::StopThinking);
                                    } else {
                                        // Normal response (for backwards compatibility)
                                        let _ = tx_clone.blocking_send(AppEvent::AddMessage(
                                            response_str,
                                            MessageType::Assistant,
                                        ));
                                    }
                                }
                            }
                            Err(_) => {
                                // Silently handle errors - don't spam stderr
                            }
                        }
                    }); // Close Python::with_gil
                } // Close while loop
            });
        }

        py.allow_threads(|| {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async move {
                // Setup terminal
                enable_raw_mode().unwrap();
                let mut stdout = io::stdout();
                execute!(stdout, EnterAlternateScreen, EnableMouseCapture).unwrap();
                let backend = CrosstermBackend::new(stdout);
                let mut terminal = Terminal::new(backend).unwrap();

                // Clear the screen
                terminal.clear().unwrap();

                // Main loop
                let mut needs_redraw = true;
                let mut last_spinner_update = std::time::Instant::now();
                loop {
                    // Draw UI if needed
                    if needs_redraw {
                        let app_ref = app.lock().unwrap();
                        terminal.draw(|f| ui::draw(f, &*app_ref)).unwrap();
                        drop(app_ref);
                        needs_redraw = false;
                    }

                    // Poll for streaming updates from Python
                    // This avoids Python calling back into Rust directly
                    let _ = callback_tx.send("__POLL_STREAMING__".to_string());

                    // Update spinner if thinking
                    {
                        let app_guard = app.lock().unwrap();
                        if app_guard.is_thinking
                            && last_spinner_update.elapsed() >= std::time::Duration::from_millis(80)
                        {
                            drop(app_guard);
                            let _ = tx.send(AppEvent::UpdateSpinner).await;
                            last_spinner_update = std::time::Instant::now();
                        } else {
                            drop(app_guard);
                        }
                    }

                    // Handle events
                    if event::poll(Duration::from_millis(50)).unwrap() {
                        match event::read().unwrap() {
                            Event::Key(key) => {
                                // Check for global quit key combination first
                                if key.modifiers.contains(KeyModifiers::CONTROL)
                                    && key.code == KeyCode::Char('q')
                                {
                                    break;
                                }

                                let mut app_guard = app.lock().unwrap();

                                // Check if we have a popup that needs a response
                                if let Some(popup) = &app_guard.popup {
                                    if popup.needs_response {
                                        match key.code {
                                            KeyCode::Char('y') | KeyCode::Char('Y') => {
                                                if let PopupType::ToolPermission {
                                                    tool_name, ..
                                                } = &popup.popup_type
                                                {
                                                    let _ = callback_tx.send(format!(
                                                        "__TOOL_PERMISSION__:{}:allow",
                                                        tool_name
                                                    ));
                                                }
                                                app_guard.close_popup();
                                                drop(app_guard);
                                                needs_redraw = true;
                                                continue;
                                            }
                                            KeyCode::Char('n') | KeyCode::Char('N') => {
                                                if let PopupType::ToolPermission {
                                                    tool_name, ..
                                                } = &popup.popup_type
                                                {
                                                    let _ = callback_tx.send(format!(
                                                        "__TOOL_PERMISSION__:{}:deny",
                                                        tool_name
                                                    ));
                                                }
                                                app_guard.close_popup();
                                                drop(app_guard);
                                                needs_redraw = true;
                                                continue;
                                            }
                                            KeyCode::Char('a') | KeyCode::Char('A') => {
                                                if let PopupType::ToolPermission {
                                                    tool_name, ..
                                                } = &popup.popup_type
                                                {
                                                    let _ = callback_tx.send(format!(
                                                        "__TOOL_PERMISSION__:{}:always_allow",
                                                        tool_name
                                                    ));
                                                }
                                                app_guard.close_popup();
                                                drop(app_guard);
                                                needs_redraw = true;
                                                continue;
                                            }
                                            KeyCode::Char('d') | KeyCode::Char('D') => {
                                                if let PopupType::ToolPermission {
                                                    tool_name, ..
                                                } = &popup.popup_type
                                                {
                                                    let _ = callback_tx.send(format!(
                                                        "__TOOL_PERMISSION__:{}:always_deny",
                                                        tool_name
                                                    ));
                                                }
                                                app_guard.close_popup();
                                                drop(app_guard);
                                                needs_redraw = true;
                                                continue;
                                            }
                                            KeyCode::Esc => {
                                                // ESC closes popup without action (defaults to deny)
                                                if let PopupType::ToolPermission {
                                                    tool_name, ..
                                                } = &popup.popup_type
                                                {
                                                    let _ = callback_tx.send(format!(
                                                        "__TOOL_PERMISSION__:{}:deny",
                                                        tool_name
                                                    ));
                                                }
                                                app_guard.close_popup();
                                                drop(app_guard);
                                                needs_redraw = true;
                                                continue;
                                            }
                                            _ => {
                                                // Ignore other keys while popup is active
                                                drop(app_guard);
                                                continue;
                                            }
                                        }
                                    }
                                }

                                // Check if we're in chat tab and handle input
                                if app_guard.selected_tab == 0 {
                                    match key.code {
                                        KeyCode::Char(c) => {
                                            app_guard.input_buffer.push(c);
                                            drop(app_guard);
                                            needs_redraw = true;
                                        }
                                        KeyCode::Backspace => {
                                            // Handle Unicode properly - remove the last grapheme cluster
                                            if !app_guard.input_buffer.is_empty() {
                                                // Find the last character boundary
                                                let mut char_boundary =
                                                    app_guard.input_buffer.len();
                                                while !app_guard
                                                    .input_buffer
                                                    .is_char_boundary(char_boundary)
                                                    && char_boundary > 0
                                                {
                                                    char_boundary -= 1;
                                                }
                                                // Find the previous character boundary
                                                if char_boundary > 0 {
                                                    char_boundary -= 1;
                                                    while !app_guard
                                                        .input_buffer
                                                        .is_char_boundary(char_boundary)
                                                        && char_boundary > 0
                                                    {
                                                        char_boundary -= 1;
                                                    }
                                                    app_guard.input_buffer.truncate(char_boundary);
                                                }
                                            }
                                            drop(app_guard);
                                            needs_redraw = true;
                                        }
                                        KeyCode::Enter => {
                                            if !app_guard.input_buffer.is_empty() {
                                                let msg = app_guard.input_buffer.clone();
                                                app_guard.input_buffer.clear();
                                                drop(app_guard);
                                                tx.send(AppEvent::SendChatMessage(msg))
                                                    .await
                                                    .unwrap();
                                            } else {
                                                drop(app_guard);
                                            }
                                        }
                                        KeyCode::Up => {
                                            // Scroll up
                                            app_guard.scroll_up();
                                            drop(app_guard);
                                            needs_redraw = true;
                                        }
                                        KeyCode::Down => {
                                            // Scroll down
                                            app_guard.scroll_down();
                                            drop(app_guard);
                                            needs_redraw = true;
                                        }
                                        KeyCode::PageUp => {
                                            // Scroll up by 10 lines
                                            for _ in 0..10 {
                                                app_guard.scroll_up();
                                            }
                                            drop(app_guard);
                                            needs_redraw = true;
                                        }
                                        KeyCode::PageDown => {
                                            // Scroll down by 10 lines
                                            for _ in 0..10 {
                                                app_guard.scroll_down();
                                            }
                                            drop(app_guard);
                                            needs_redraw = true;
                                        }
                                        KeyCode::Home => {
                                            // Scroll to top
                                            app_guard.scroll_offset = 0;
                                            drop(app_guard);
                                            needs_redraw = true;
                                        }
                                        KeyCode::End => {
                                            // Scroll to bottom (max value, will be clamped in draw)
                                            app_guard.scroll_offset = usize::MAX;
                                            drop(app_guard);
                                            needs_redraw = true;
                                        }
                                        KeyCode::Tab => {
                                            // Tab does nothing in chat
                                            drop(app_guard);
                                        }
                                        KeyCode::Esc => {
                                            // Check if Wake is processing
                                            drop(app_guard);
                                            // Send interrupt command
                                            let _ = callback_tx.send("__INTERRUPT__".to_string());
                                        }
                                        _ => {
                                            drop(app_guard);
                                        }
                                    }
                                } else {
                                    // Non-chat tab navigation
                                    // Non-chat tabs removed - should never get here
                                    drop(app_guard);
                                }
                            }
                            Event::Mouse(mouse) => match mouse.kind {
                                event::MouseEventKind::ScrollDown => {
                                    let mut app_guard = app.lock().unwrap();
                                    app_guard.scroll_down();
                                    drop(app_guard);
                                    needs_redraw = true;
                                }
                                event::MouseEventKind::ScrollUp => {
                                    let mut app_guard = app.lock().unwrap();
                                    app_guard.scroll_up();
                                    drop(app_guard);
                                    needs_redraw = true;
                                }
                                _ => {}
                            },
                            _ => {}
                        }
                    }

                    // Process app events
                    if let Ok(event) = rx.try_recv() {
                        match event {
                            AppEvent::Quit => break,
                            AppEvent::UpdateStats(stats) => {
                                let mut app = app.lock().unwrap();
                                app.update_stats(stats);
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::AddMessage(msg, msg_type) => {
                                let mut app = app.lock().unwrap();
                                // Check if we're at the bottom before adding the message
                                let was_at_bottom =
                                    app.scroll_offset == 0 || app.scroll_offset >= 1000000;
                                app.add_message(msg, msg_type);
                                // Auto-scroll only if we were already at the bottom
                                if was_at_bottom {
                                    app.scroll_offset = 0; // Will be adjusted in UI
                                }
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::AddStreamingText(text) => {
                                let mut app = app.lock().unwrap();
                                app.add_streaming_text(text);
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::AddSystemMessage(msg) => {
                                // Show system messages - they contain important info
                                let mut app = app.lock().unwrap();
                                app.add_system_message(msg);
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::SendChatMessage(msg) => {
                                // Add user message and clear system buffer
                                {
                                    let mut app = app.lock().unwrap();
                                    app.add_message(msg.clone(), MessageType::User);
                                    app.clear_system_buffer();
                                }
                                needs_redraw = true;
                                // Send to Python callback
                                let _ = callback_tx.send(msg);
                            }
                            AppEvent::Refresh => {
                                needs_redraw = true;
                            }
                            AppEvent::StartThinking => {
                                let mut app = app.lock().unwrap();
                                app.start_thinking();
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::StopThinking => {
                                let mut app = app.lock().unwrap();
                                app.stop_thinking();
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::UpdateSpinner => {
                                let mut app = app.lock().unwrap();
                                app.update_spinner();
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::UpdateTokenCount(count) => {
                                let mut app = app.lock().unwrap();
                                app.current_tokens = count;
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::ShowToolPermission(tool_name, tool_input) => {
                                let mut app = app.lock().unwrap();
                                app.show_tool_permission(tool_name, tool_input);
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::ShowToolUse(tool_name) => {
                                let mut app = app.lock().unwrap();
                                app.show_tool_use(tool_name);
                                drop(app);
                                needs_redraw = true;
                            }
                            AppEvent::ClosePopup => {
                                let mut app = app.lock().unwrap();
                                app.close_popup();
                                drop(app);
                                needs_redraw = true;
                            }
                            _ => {}
                        }
                    }
                }

                // Restore terminal
                disable_raw_mode().unwrap();
                execute!(
                    terminal.backend_mut(),
                    LeaveAlternateScreen,
                    DisableMouseCapture
                )
                .unwrap();
                terminal.show_cursor().unwrap();
            });
        });
        Ok(())
    }

    fn update_stats(&self, stats: Stats) -> PyResult<()> {
        if let Some(tx) = &self.tx {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime
                .block_on(tx.send(AppEvent::UpdateStats(stats)))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    fn add_message(&self, message: String, msg_type: String) -> PyResult<()> {
        let msg_type = match msg_type.as_str() {
            "user" => MessageType::User,
            "assistant" => MessageType::Assistant,
            "system" => MessageType::System,
            _ => MessageType::System,
        };

        if let Some(tx) = &self.tx {
            tx.blocking_send(AppEvent::AddMessage(message, msg_type))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    fn refresh(&self) -> PyResult<()> {
        if let Some(tx) = &self.tx {
            tx.blocking_send(AppEvent::Refresh)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }

    fn quit(&self) -> PyResult<()> {
        if let Some(tx) = &self.tx {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime
                .block_on(tx.send(AppEvent::Quit))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        Ok(())
    }
}

/// Python module definition
#[pymodule]
fn bicamrl_tui(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BicamrlTUI>()?;
    m.add_class::<Stats>()?;
    Ok(())
}
