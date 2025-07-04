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
mod app_core;
mod code_highlighter;
mod debug_logger;
mod markdown;
mod spinner;
mod theme;
mod ui;
mod unicode_utils;
mod widgets;

use app::PopupType;
use app_core::AppCore;
use debug_logger::init_debug_log;

#[pyclass]
pub struct BicamrlTUI {
    core: AppCore,
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
        Ok(Self {
            core: AppCore::new(),
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

        let core = self.core.clone();
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
                                            // Use AppCore to parse the update
                                            if let Some(event) =
                                                AppCore::parse_streaming_update(json_str)
                                            {
                                                debug_log!("[CALLBACK THREAD] Sending event to main loop: {:?}", event);
                                                match tx_clone.blocking_send(event) {
                                                    Ok(_) => debug_log!("[CALLBACK THREAD] Event sent successfully"),
                                                    Err(e) => debug_log!("[CALLBACK THREAD] Failed to send event: {}", e),
                                                }
                                            } else {
                                                debug_log!("[CALLBACK THREAD] AppCore returned None for JSON: {}", json_str);
                                            }
                                        }
                                    } else if response_str == "__NO_STREAMING__" {
                                        // No streaming updates available
                                    } else if response_str.starts_with("__STATS__:") {
                                        // Parse stats update
                                        if let Some(json_str) =
                                            response_str.strip_prefix("__STATS__:")
                                        {
                                            if let Some(stats) =
                                                AppCore::parse_stats_update(json_str)
                                            {
                                                let _ = tx_clone
                                                    .blocking_send(AppEvent::UpdateStats(stats));
                                            }
                                        }
                                    } else if response_str == "__NO_STATS__" {
                                        // No stats updates available
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
            let core = core.clone();
            let app = core.get_app();

            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async move {
                // Initialize debug logging
                init_debug_log();
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
                let mut last_stats_poll = std::time::Instant::now();
                loop {
                    // Draw UI if needed
                    if needs_redraw {
                        let app_ref = app.lock().unwrap();
                        let has_popup = app_ref.popup.is_some();
                        terminal.draw(|f| ui::draw(f, &*app_ref)).unwrap();
                        drop(app_ref);
                        if has_popup {
                            debug_log!("[MAIN LOOP] Drew frame with popup visible");
                        }
                        needs_redraw = false;
                    }

                    // Poll for streaming updates from Python
                    // This avoids Python calling back into Rust directly
                    let _ = callback_tx.send("__POLL_STREAMING__".to_string());

                    // Poll for stats updates periodically
                    if last_stats_poll.elapsed() >= std::time::Duration::from_millis(500) {
                        let _ = callback_tx.send("__POLL_STATS__".to_string());
                        last_stats_poll = std::time::Instant::now();
                    }

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
                        // Log important events
                        match &event {
                            AppEvent::ShowToolUse(tool) => {
                                debug_log!("[MAIN LOOP] Received ShowToolUse event for: {}", tool);
                            }
                            AppEvent::ClosePopup => {
                                debug_log!("[MAIN LOOP] Received ClosePopup event");
                            }
                            _ => {}
                        }

                        match &event {
                            AppEvent::Quit => break,
                            AppEvent::SendChatMessage(msg) => {
                                // Special handling for chat messages
                                {
                                    let mut app = app.lock().unwrap();
                                    app.add_message(msg.clone(), MessageType::User);
                                    app.clear_system_buffer();
                                }
                                needs_redraw = true;
                                // Send to Python callback
                                let _ = callback_tx.send(msg.clone());
                            }
                            _ => {
                                // Use AppCore to process all other events
                                needs_redraw = core.process_event(event) || needs_redraw;
                            }
                        }
                    }
                }

                // Log final state before exiting
                {
                    let app = app.lock().unwrap();
                    debug_log!("=== TUI Exit Summary ===");
                    debug_log!(
                        "Popup events - shows: {}, closes: {}",
                        app.popup_events.0,
                        app.popup_events.1
                    );
                    debug_log!("Final popup state: {:?}", app.popup.is_some());
                    debug_log!("Total messages: {}", app.messages.len());
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
