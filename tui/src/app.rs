use crate::spinner::SpinnerState;
use crate::{MessageType, Stats};
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
        self.current_tokens = 0;
    }

    pub fn update_spinner(&mut self) {
        if self.is_thinking {
            self.spinner_state.update();
        }
    }

    pub fn show_popup(&mut self, popup: Popup) {
        self.popup = Some(popup);
    }

    pub fn close_popup(&mut self) {
        self.popup = None;
    }

    pub fn show_tool_permission(&mut self, tool_name: String, tool_input: String) {
        // Split long input into multiple lines if needed
        let mut content = vec![
            format!("Tool: {}", tool_name),
            "".to_string(),
            "Input:".to_string(),
        ];

        // Split tool input by lines and limit line length
        for line in tool_input.lines() {
            if line.len() > 50 {
                // Wrap long lines
                for chunk in line.chars().collect::<Vec<_>>().chunks(50) {
                    content.push(chunk.iter().collect());
                }
            } else {
                content.push(line.to_string());
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
