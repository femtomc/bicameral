use crate::spinner::SpinnerState;
use crate::{debug_log, MessageType, Stats};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct Message {
    pub content: String,
    pub msg_type: MessageType,
    pub timestamp: chrono::DateTime<chrono::Local>,
}

#[derive(Debug)]
pub struct App {
    pub state: AppState,
    pub stats: Stats,
    pub messages: VecDeque<Message>,
    pub selected_tab: usize,
    pub input_buffer: String,
    pub scroll_offset: usize,
    pub system_buffer: VecDeque<String>,
    pub is_thinking: bool,
    pub spinner_state: SpinnerState,
    pub current_tokens: usize,
    pub popup: Option<Popup>,
    pub popup_shown_at: Option<std::time::Instant>,
    pub popup_events: (usize, usize), // (show_count, close_count)
}

#[derive(Debug, Clone)]
pub struct Popup {
    pub title: String,
    pub content: Vec<String>,
    pub popup_type: PopupType,
    pub needs_response: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PopupType {
    ToolPermission {
        tool_name: String,
        tool_input: String,
    },
    ToolUse {
        tool_name: String,
    },
    Info,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppState {
    Running,
    Quitting,
}

impl App {
    pub fn new() -> Self {
        Self {
            state: AppState::Running,
            stats: Stats::new(),
            messages: VecDeque::with_capacity(1000),
            selected_tab: 0,
            input_buffer: String::new(),
            scroll_offset: 0,
            system_buffer: VecDeque::with_capacity(100),
            is_thinking: false,
            spinner_state: SpinnerState::new(),
            current_tokens: 0,
            popup: None,
            popup_shown_at: None,
            popup_events: (0, 0),
        }
    }

    pub fn update_stats(&mut self, stats: Stats) {
        self.stats = stats;
    }

    pub fn add_message(&mut self, content: String, msg_type: MessageType) {
        let message = Message {
            content,
            msg_type,
            timestamp: chrono::Local::now(),
        };
        self.messages.push_back(message);

        // Keep only last 1000 messages
        if self.messages.len() > 1000 {
            self.messages.pop_front();
        }

        // Don't auto-scroll - let the UI handle it
        // The UI will auto-scroll only if the user is already at the bottom
    }

    pub fn add_streaming_text(&mut self, text: String) {
        // Add to the last assistant message if it exists and is incomplete
        if let Some(last_msg) = self.messages.back_mut() {
            if matches!(last_msg.msg_type, MessageType::Assistant) {
                // Append to existing message
                last_msg.content.push_str(&text);
                return;
            }
        }

        // Otherwise create a new assistant message
        self.add_message(text, MessageType::Assistant);
    }

    pub fn remove_last_message(&mut self) {
        self.messages.pop_back();
    }

    pub fn next_tab(&mut self) {
        // Only one tab now
    }

    pub fn previous_tab(&mut self) {
        // Only one tab now
    }

    pub fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }

    pub fn scroll_down(&mut self) {
        self.scroll_offset += 1;
    }

    pub fn get_total_lines(&self) -> usize {
        // This will be calculated by the UI when rendering
        // For now, return a large number to allow unlimited scrolling
        usize::MAX
    }

    pub fn quit(&mut self) {
        self.state = AppState::Quitting;
    }

    pub fn add_system_message(&mut self, message: String) {
        self.system_buffer.push_back(message);
        // Keep only last 100 system messages
        if self.system_buffer.len() > 100 {
            self.system_buffer.pop_front();
        }
    }

    pub fn clear_system_buffer(&mut self) {
        self.system_buffer.clear();
    }

    pub fn start_thinking(&mut self) {
        self.is_thinking = true;
        self.spinner_state.reset();
        self.current_tokens = 0;
    }

    pub fn stop_thinking(&mut self) {
        self.is_thinking = false;
        // Don't clear current_tokens here - we want to keep the count
    }

    pub fn update_spinner(&mut self) {
        if self.is_thinking {
            self.spinner_state.update();
        }
    }

    pub fn show_popup(&mut self, popup: Popup) {
        let popup_type = format!("{:?}", popup.popup_type);
        self.popup = Some(popup);
        self.popup_shown_at = Some(std::time::Instant::now());
        self.popup_events.0 += 1;
        debug_log!(
            "[APP] Popup shown: {}. Total shows: {}, closes: {}",
            popup_type,
            self.popup_events.0,
            self.popup_events.1
        );
    }

    pub fn close_popup(&mut self) {
        // Just close the popup - no timing checks
        let was_open = self.popup.is_some();
        self.popup = None;
        self.popup_shown_at = None;
        self.popup_events.1 += 1;
        debug_log!(
            "[APP] Popup close called (was_open: {}). Total shows: {}, closes: {}",
            was_open,
            self.popup_events.0,
            self.popup_events.1
        );
    }

    pub fn show_tool_permission(&mut self, tool_name: String, tool_input: String) {
        // Split long input into multiple lines if needed
        let mut content = vec![format!("Tool: {}", tool_name), "".to_string()];

        // Try to parse the JSON and format it nicely
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&tool_input) {
            // Format each field of the JSON
            if let Some(obj) = json_value.as_object() {
                for (key, value) in obj {
                    // Format the key
                    content.push(format!("{}:", key.to_uppercase()));

                    // Format the value
                    let value_str = match value {
                        serde_json::Value::String(s) => s.clone(),
                        _ => value.to_string(),
                    };

                    // Wrap long lines, indented
                    if value_str.len() > 60 {
                        for chunk in value_str.chars().collect::<Vec<_>>().chunks(60) {
                            content.push(format!("  {}", chunk.iter().collect::<String>()));
                        }
                    } else {
                        content.push(format!("  {}", value_str));
                    }
                    content.push("".to_string());
                }
            } else {
                // Not an object, just display as-is but wrapped
                content.push("Input:".to_string());
                for line in tool_input.lines() {
                    if line.len() > 50 {
                        for chunk in line.chars().collect::<Vec<_>>().chunks(50) {
                            content.push(chunk.iter().collect());
                        }
                    } else {
                        content.push(line.to_string());
                    }
                }
            }
        } else {
            // Not JSON, display as plain text
            content.push("Input:".to_string());
            for line in tool_input.lines() {
                if line.len() > 50 {
                    for chunk in line.chars().collect::<Vec<_>>().chunks(50) {
                        content.push(chunk.iter().collect());
                    }
                } else {
                    content.push(line.to_string());
                }
            }
        }

        content.extend(vec![
            "".to_string(),
            "Allow this tool to run?".to_string(),
            "".to_string(),
            "Press:".to_string(),
            "  Y - Allow this time".to_string(),
            "  N - Deny this time".to_string(),
            "  A - Always allow this tool".to_string(),
            "  D - Always deny this tool".to_string(),
        ]);

        let popup = Popup {
            title: format!("⚠️  Permission Required: {}", tool_name),
            content,
            popup_type: PopupType::ToolPermission {
                tool_name,
                tool_input,
            },
            needs_response: true,
        };
        self.show_popup(popup);
    }

    pub fn show_tool_use(&mut self, tool_name: String) {
        let popup = Popup {
            title: "Tool Running".to_string(),
            content: vec![
                format!("Executing: {}", tool_name),
                "".to_string(),
                "Please wait...".to_string(),
            ],
            popup_type: PopupType::ToolUse { tool_name },
            needs_response: false,
        };
        self.show_popup(popup);
    }
}
