use crate::app::App;
use crate::{debug_log, AppEvent, MessageType, Stats};
use std::sync::{Arc, Mutex};

/// Core application logic separated from terminal rendering
/// This can be fully tested without any terminal interaction
#[derive(Clone)]
pub struct AppCore {
    app: Arc<Mutex<App>>,
}

impl AppCore {
    pub fn new() -> Self {
        Self {
            app: Arc::new(Mutex::new(App::new())),
        }
    }

    pub fn get_app(&self) -> Arc<Mutex<App>> {
        self.app.clone()
    }

    /// Process an app event and update state
    /// Returns true if the UI needs to be redrawn
    pub fn process_event(&self, event: AppEvent) -> bool {
        match event {
            AppEvent::UpdateStats(stats) => {
                let mut app = self.app.lock().unwrap();
                app.update_stats(stats);
                true
            }
            AppEvent::AddMessage(msg, msg_type) => {
                let mut app = self.app.lock().unwrap();
                app.add_message(msg, msg_type);
                true
            }
            AppEvent::AddStreamingText(text) => {
                let mut app = self.app.lock().unwrap();
                app.add_streaming_text(text);
                true
            }
            AppEvent::AddSystemMessage(msg) => {
                let mut app = self.app.lock().unwrap();
                app.add_system_message(msg);
                true
            }
            AppEvent::StartThinking => {
                let mut app = self.app.lock().unwrap();
                app.start_thinking();
                true
            }
            AppEvent::StopThinking => {
                let mut app = self.app.lock().unwrap();
                app.stop_thinking();
                true
            }
            AppEvent::UpdateSpinner => {
                let mut app = self.app.lock().unwrap();
                app.update_spinner();
                true
            }
            AppEvent::UpdateTokenCount(count) => {
                let mut app = self.app.lock().unwrap();
                app.current_tokens = count;
                true
            }
            AppEvent::ShowToolPermission(tool_name, tool_input) => {
                let mut app = self.app.lock().unwrap();
                app.show_tool_permission(tool_name, tool_input);
                true
            }
            AppEvent::ShowToolUse(tool_name) => {
                debug_log!("[AppCore::process_event] ShowToolUse for: {}", tool_name);
                let mut app = self.app.lock().unwrap();
                app.show_tool_use(tool_name);
                true
            }
            AppEvent::ClosePopup => {
                debug_log!("[AppCore::process_event] ClosePopup - calling app.close_popup()");
                let mut app = self.app.lock().unwrap();
                let had_popup = app.popup.is_some();
                app.close_popup();
                debug_log!(
                    "[AppCore::process_event] ClosePopup - had_popup: {}, has_popup: {}",
                    had_popup,
                    app.popup.is_some()
                );
                true // Always redraw after closing popup
            }
            _ => false,
        }
    }

    /// Parse a streaming update from Python and convert to AppEvent
    pub fn parse_streaming_update(json_str: &str) -> Option<AppEvent> {
        // Only log important events to reduce noise
        if json_str.contains("tool") || json_str.contains("CLOSE_POPUP") {
            debug_log!("[AppCore::parse_streaming_update] Raw JSON: {}", json_str);
        }

        if let Ok(update) = serde_json::from_str::<serde_json::Value>(json_str) {
            let update_type = update.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let content = update
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let is_complete = update
                .get("is_complete")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            debug_log!(
                "[AppCore::parse_streaming_update] Parsed - type: '{}', content: '{}'",
                update_type,
                content
            );

            match update_type {
                "text" => {
                    if is_complete && !content.is_empty() {
                        Some(AppEvent::AddMessage(content, MessageType::Assistant))
                    } else if !content.is_empty() {
                        Some(AppEvent::AddStreamingText(content))
                    } else {
                        None
                    }
                }
                "system" => {
                    if content.starts_with("__UPDATE_TOKENS__:") {
                        if let Some(token_str) = content.strip_prefix("__UPDATE_TOKENS__:") {
                            if let Ok(tokens) = token_str.parse::<usize>() {
                                Some(AppEvent::UpdateTokenCount(tokens))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else if content.starts_with("__TOOL_PERMISSION_REQUEST__:") {
                        if let Some(perm_str) = content.strip_prefix("__TOOL_PERMISSION_REQUEST__:")
                        {
                            let parts: Vec<&str> = perm_str.splitn(2, ':').collect();
                            if parts.len() >= 1 {
                                let tool_name = parts[0].to_string();
                                let tool_input = if parts.len() > 1 {
                                    if let Ok(parsed) =
                                        serde_json::from_str::<serde_json::Value>(parts[1])
                                    {
                                        serde_json::to_string_pretty(&parsed)
                                            .unwrap_or_else(|_| parts[1].to_string())
                                    } else {
                                        parts[1].to_string()
                                    }
                                } else {
                                    String::new()
                                };
                                Some(AppEvent::ShowToolPermission(tool_name, tool_input))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else if content == "__CLOSE_POPUP__" {
                        debug_log!("[AppCore::parse_streaming_update] Returning ClosePopup event");
                        Some(AppEvent::ClosePopup)
                    } else if !content.is_empty() {
                        Some(AppEvent::AddSystemMessage(content))
                    } else {
                        None
                    }
                }
                "usage" => {
                    if !content.is_empty() {
                        Some(AppEvent::AddSystemMessage(content))
                    } else {
                        None
                    }
                }
                "tool_use" => {
                    debug_log!(
                        "[AppCore::parse_streaming_update] tool_use type, content: '{}'",
                        content
                    );
                    if content.starts_with("Using tool: ") {
                        if let Some(tool_name) = content.strip_prefix("Using tool: ") {
                            debug_log!("[AppCore::parse_streaming_update] Extracted tool name: '{}', returning ShowToolUse", tool_name);
                            Some(AppEvent::ShowToolUse(tool_name.to_string()))
                        } else {
                            None
                        }
                    } else {
                        debug_log!("[AppCore::parse_streaming_update] Content doesn't start with 'Using tool: '");
                        None
                    }
                }
                "tool_result" => {
                    debug_log!(
                        "[AppCore::parse_streaming_update] tool_result type - returning ClosePopup"
                    );
                    Some(AppEvent::ClosePopup)
                }
                _ => {
                    if !content.is_empty() {
                        Some(AppEvent::AddStreamingText(content))
                    } else {
                        None
                    }
                }
            }
        } else {
            debug_log!(
                "[AppCore::parse_streaming_update] Failed to parse JSON: {}",
                json_str
            );
            None
        }
    }

    /// Parse a stats update from Python
    pub fn parse_stats_update(json_str: &str) -> Option<Stats> {
        if let Ok(stats_data) = serde_json::from_str::<serde_json::Value>(json_str) {
            Some(Stats {
                total_interactions: stats_data
                    .get("total_interactions")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                total_patterns: stats_data
                    .get("total_patterns")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                success_rate: stats_data
                    .get("success_rate")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                active_memories: stats_data
                    .get("active_memories")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                tokens_used: stats_data
                    .get("tokens_used")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                active_sessions: stats_data
                    .get("active_sessions")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_use() {
        let json =
            r#"{"type": "tool_use", "content": "Using tool: read_file", "is_complete": false}"#;
        let event = AppCore::parse_streaming_update(json);

        match event {
            Some(AppEvent::ShowToolUse(tool_name)) => {
                assert_eq!(tool_name, "read_file");
            }
            _ => panic!("Expected ShowToolUse event"),
        }
    }

    #[test]
    fn test_parse_close_popup() {
        let json = r#"{"type": "system", "content": "__CLOSE_POPUP__", "is_complete": false}"#;
        let event = AppCore::parse_streaming_update(json);

        match event {
            Some(AppEvent::ClosePopup) => {
                // Success
            }
            _ => panic!("Expected ClosePopup event"),
        }
    }

    #[test]
    fn test_process_events() {
        let core = AppCore::new();

        // Test ShowToolUse
        let needs_redraw = core.process_event(AppEvent::ShowToolUse("test_tool".to_string()));
        assert!(needs_redraw);

        {
            let app = core.app.lock().unwrap();
            assert!(app.popup.is_some());
        }

        // Test ClosePopup
        let needs_redraw = core.process_event(AppEvent::ClosePopup);
        assert!(needs_redraw);

        {
            let app = core.app.lock().unwrap();
            assert!(app.popup.is_none());
        }
    }
}
