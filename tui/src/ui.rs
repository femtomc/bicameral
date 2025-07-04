use crate::app::{App, Popup, PopupType};
use crate::markdown::parse_markdown;
use crate::theme::AndromedaTheme;
use crate::unicode_utils::display_width;
use crate::MessageType;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

pub fn draw(f: &mut Frame, app: &App) {
    // Fill the entire screen with the background color
    let background = Block::default().style(Style::default().bg(AndromedaTheme::BACKGROUND));
    f.render_widget(background, f.size());

    // Just draw the content area - no footer
    draw_content(f, f.size(), app);

    // Draw popup if active
    if let Some(popup) = &app.popup {
        draw_popup(f, popup);
    }
}

fn draw_content(f: &mut Frame, area: Rect, app: &App) {
    draw_chat(f, area, app);
}

fn wrap_message_lines(
    spans: Vec<Span>,
    line_number: &mut usize,
    max_width: usize,
) -> Vec<Line<'static>> {
    let mut result = Vec::new();

    // Calculate prefix width (line number + separator)
    let prefix_width = 4 + 1 + 2 + 2; // "9999 │  " = 9 chars
    let content_width = max_width.saturating_sub(prefix_width);

    // Join all spans into a single text
    let full_text: String = spans.iter().map(|s| s.content.as_ref()).collect();

    // Simple word wrapping
    let mut current_line = String::new();
    let mut current_width = 0;

    for word in full_text.split_whitespace() {
        let word_width = display_width(word);

        if current_width > 0 && current_width + word_width + 1 > content_width {
            // Emit current line
            let line_spans = vec![
                Span::styled(
                    format!("{:4} ", line_number),
                    Style::default().fg(AndromedaTheme::COMMENT),
                ),
                Span::styled("│ ", Style::default().fg(AndromedaTheme::BORDER)),
                Span::raw("  "),
                Span::raw(current_line.clone()),
            ];
            result.push(Line::from(line_spans));
            *line_number += 1;

            current_line = word.to_string();
            current_width = word_width;
        } else {
            if !current_line.is_empty() {
                current_line.push(' ');
                current_width += 1;
            }
            current_line.push_str(word);
            current_width += word_width;
        }
    }

    // Emit final line
    if !current_line.is_empty() {
        let line_spans = vec![
            Span::styled(
                format!("{:4} ", line_number),
                Style::default().fg(AndromedaTheme::COMMENT),
            ),
            Span::styled("│ ", Style::default().fg(AndromedaTheme::BORDER)),
            Span::raw("  "),
            Span::raw(current_line),
        ];
        result.push(Line::from(line_spans));
        *line_number += 1;
    }

    result
}

fn draw_chat(f: &mut Frame, area: Rect, app: &App) {
    // Layout with system status bar
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // Messages
            Constraint::Length(3), // Input
            Constraint::Length(2), // System status
        ])
        .split(area);

    // Draw messages
    if app.messages.is_empty() {
        // Show empty state
        let empty_text = "No messages yet. Type something and press Enter!";
        let empty_widget = Paragraph::new(empty_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Chat")
                    .border_style(Style::default().fg(AndromedaTheme::BORDER))
                    .title_style(Style::default().fg(AndromedaTheme::TEAL))
                    .style(Style::default().bg(AndromedaTheme::BACKGROUND)),
            )
            .style(Style::default().fg(AndromedaTheme::COMMENT))
            .alignment(Alignment::Center);
        f.render_widget(empty_widget, chunks[0]);
    } else {
        // Convert messages to text lines with markdown parsing and line numbers
        let mut all_lines = Vec::new();
        let mut line_number = 1;

        for msg in app.messages.iter() {
            let style = match msg.msg_type {
                MessageType::User => Style::default().fg(AndromedaTheme::USER_MSG),
                MessageType::Assistant => Style::default().fg(AndromedaTheme::ASSISTANT_MSG),
                MessageType::System => Style::default().fg(AndromedaTheme::SYSTEM_MSG),
            };

            let prefix = match msg.msg_type {
                MessageType::User => "You: ",
                MessageType::Assistant => "Wake: ",
                MessageType::System => "System: ",
            };

            // Add prefix line with line number
            let prefix_spans = vec![
                Span::styled(
                    format!("{:4} ", line_number),
                    Style::default().fg(AndromedaTheme::COMMENT),
                ),
                Span::styled("│ ", Style::default().fg(AndromedaTheme::BORDER)),
                Span::styled(prefix, style.add_modifier(Modifier::BOLD)),
            ];
            all_lines.push(Line::from(prefix_spans));
            line_number += 1;

            // Parse markdown content and wrap lines
            let content_lines = parse_markdown(&msg.content);
            let area_width = chunks[0].width as usize;

            for line in content_lines {
                // Wrap each line to fit within the available width
                let wrapped_lines = wrap_message_lines(line.spans, &mut line_number, area_width);
                all_lines.extend(wrapped_lines);
            }

            // Add empty line between messages
            all_lines.push(Line::from(vec![
                Span::styled(
                    format!("{:4} ", line_number),
                    Style::default().fg(AndromedaTheme::COMMENT),
                ),
                Span::styled("│ ", Style::default().fg(AndromedaTheme::BORDER)),
            ]));
            line_number += 1;
        }

        // Add thinking animation as the last line if active
        if app.is_thinking {
            let spinner_frame = app.spinner_state.current_frame();
            let thinking_phrase = app.spinner_state.current_phrase();
            let elapsed = app.spinner_state.elapsed_secs;

            let mut thinking_spans = vec![
                Span::styled(
                    format!("{:4} ", line_number),
                    Style::default().fg(AndromedaTheme::COMMENT),
                ),
                Span::styled("│ ", Style::default().fg(AndromedaTheme::BORDER)),
                Span::styled(
                    "Wake: ",
                    Style::default()
                        .fg(AndromedaTheme::ASSISTANT_MSG)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(spinner_frame, Style::default().fg(AndromedaTheme::TEAL)),
                Span::raw(" "),
                Span::styled(thinking_phrase, Style::default().fg(AndromedaTheme::PURPLE)),
            ];

            // Add token count if available
            if app.current_tokens > 0 {
                thinking_spans.push(Span::raw(" • "));
                thinking_spans.push(Span::styled(
                    format!("{} tokens", app.current_tokens),
                    Style::default().fg(AndromedaTheme::YELLOW),
                ));
            }

            // Add elapsed time
            thinking_spans.push(Span::raw(" • "));
            thinking_spans.push(Span::styled(
                format!("{}s", elapsed),
                Style::default().fg(AndromedaTheme::COMMENT),
            ));

            let thinking_line = Line::from(thinking_spans);
            all_lines.push(thinking_line);
        }

        // Calculate visible area
        let visible_height = chunks[0].height.saturating_sub(2) as usize; // Subtract 2 for borders
        let total_lines = all_lines.len();

        // Calculate maximum scroll offset that still shows content
        let max_scroll = if total_lines > visible_height {
            total_lines.saturating_sub(visible_height) // Show last visible_height lines
        } else {
            0
        };

        // Apply scroll offset with proper bounds
        // If scroll_offset is very large (from old auto-scroll), clamp to max
        let scroll_offset = if app.scroll_offset > 1000000 {
            max_scroll // Jump to bottom
        } else {
            app.scroll_offset.min(max_scroll)
        };

        let start_line = scroll_offset;
        let end_line = (start_line + visible_height).min(total_lines);

        // Get visible lines
        let visible_lines: Vec<Line> = if total_lines > 0 {
            all_lines
                .into_iter()
                .skip(start_line)
                .take(visible_height.min(total_lines - start_line)) // Don't take more than available
                .collect()
        } else {
            Vec::new()
        };

        // Add scroll indicator to title
        let scroll_indicator = if total_lines > visible_height {
            format!(" [{}-{}/{}]", start_line + 1, end_line, total_lines)
        } else {
            String::new()
        };

        let messages_widget = Paragraph::new(visible_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(
                        "Chat ({} messages){}",
                        app.messages.len(),
                        scroll_indicator
                    ))
                    .border_style(Style::default().fg(AndromedaTheme::BORDER))
                    .title_style(Style::default().fg(AndromedaTheme::TEAL))
                    .style(Style::default().bg(AndromedaTheme::BACKGROUND)),
            )
            .style(Style::default().fg(AndromedaTheme::FOREGROUND));

        f.render_widget(messages_widget, chunks[0]);
    }

    // Draw input
    let input = Paragraph::new(app.input_buffer.as_str())
        .style(Style::default().fg(AndromedaTheme::INPUT_TEXT))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Input")
                .border_style(Style::default().fg(AndromedaTheme::BORDER))
                .title_style(Style::default().fg(AndromedaTheme::YELLOW))
                .style(Style::default().bg(AndromedaTheme::BACKGROUND)),
        );
    f.render_widget(input, chunks[1]);

    // Draw system status bar with stats
    let popup_status = if app.popup.is_some() {
        " | POPUP ACTIVE"
    } else {
        ""
    };

    let system_text = if app.system_buffer.is_empty() {
        // Show stats when no system messages
        format!(
            "Ready | Interactions: {} | Patterns: {} | Tokens: {} | Sessions: {} | Events: show={}/close={}{}",
            app.stats.total_interactions,
            app.stats.total_patterns,
            app.stats.tokens_used,
            app.stats.active_sessions,
            app.popup_events.0,
            app.popup_events.1,
            popup_status
        )
    } else {
        // Show the most recent system message with token count
        let msg = app.system_buffer.back().cloned().unwrap_or_default();
        format!(
            "{} | Tokens: {} | Events: show={}/close={}{}",
            msg, app.stats.tokens_used, app.popup_events.0, app.popup_events.1, popup_status
        )
    };

    let system_status = Paragraph::new(system_text)
        .style(Style::default().fg(AndromedaTheme::ORANGE))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Status")
                .border_style(Style::default().fg(AndromedaTheme::BORDER))
                .title_style(Style::default().fg(AndromedaTheme::PURPLE))
                .style(Style::default().bg(AndromedaTheme::BACKGROUND)),
        );
    f.render_widget(system_status, chunks[2]);
}

fn draw_popup(f: &mut Frame, popup: &Popup) {
    // Calculate popup dimensions
    let area = f.size();
    let popup_width = 60.min(area.width - 4);
    let popup_height = (popup.content.len() as u16 + 4).min(area.height - 4);

    // Center the popup
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;

    let popup_area = Rect {
        x,
        y,
        width: popup_width,
        height: popup_height,
    };

    // Clear the area behind the popup
    f.render_widget(Clear, popup_area);

    // Determine popup style based on type
    let (border_color, title_color) = match &popup.popup_type {
        PopupType::ToolPermission { .. } => (AndromedaTheme::YELLOW, AndromedaTheme::YELLOW),
        PopupType::ToolUse { .. } => (AndromedaTheme::BLUE, AndromedaTheme::BLUE),
        PopupType::Info => (AndromedaTheme::GREEN, AndromedaTheme::GREEN),
        PopupType::Error => (AndromedaTheme::RED, AndromedaTheme::RED),
    };

    // Create popup content
    let content_lines: Vec<Line> = popup
        .content
        .iter()
        .map(|line| Line::from(vec![Span::raw(line.clone())]))
        .collect();

    let popup_widget = Paragraph::new(content_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(popup.title.clone())
                .border_style(
                    Style::default()
                        .fg(border_color)
                        .add_modifier(Modifier::BOLD),
                )
                .title_style(
                    Style::default()
                        .fg(title_color)
                        .add_modifier(Modifier::BOLD),
                )
                .style(Style::default().bg(AndromedaTheme::BACKGROUND)),
        )
        .style(Style::default().fg(AndromedaTheme::FOREGROUND))
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });

    f.render_widget(popup_widget, popup_area);
}
