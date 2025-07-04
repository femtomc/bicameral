use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
// Unicode width available if needed
// use unicode_width::UnicodeWidthStr;
use crate::code_highlighter::CodeHighlighter;

/// Parse markdown text and convert to styled ratatui Lines
pub fn parse_markdown(text: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut current_line_spans = Vec::new();
    let mut in_code_block = false;
    let mut code_language: Option<&str> = None;
    let mut code_lines: Vec<String> = Vec::new();
    let highlighter = CodeHighlighter::new();

    for line in text.lines() {
        // Check for code block delimiters
        if line.starts_with("```") {
            if in_code_block {
                // End of code block - render it
                in_code_block = false;

                // Add code block with background
                for code_line in &code_lines {
                    let highlighted = highlighter.highlight_line(code_line, code_language);
                    lines.push(Line::from(highlighted));
                }
                code_lines.clear();
                code_language = None;
            } else {
                // Start of code block
                in_code_block = true;
                let lang = line.trim_start_matches("```").trim();
                code_language = if lang.is_empty() { None } else { Some(lang) };
            }
            continue;
        }

        if in_code_block {
            code_lines.push(line.to_string());
            continue;
        }
        if line.is_empty() {
            if !current_line_spans.is_empty() {
                lines.push(Line::from(current_line_spans.clone()));
                current_line_spans.clear();
            }
            lines.push(Line::from(""));
            continue;
        }

        // Parse headers
        if let Some(header) = parse_header(line) {
            lines.push(header);
            continue;
        }

        // Parse list items
        if let Some(list_item) = parse_list_item(line) {
            lines.push(list_item);
            continue;
        }

        // Parse inline elements
        current_line_spans = parse_inline(line);
        lines.push(Line::from(current_line_spans.clone()));
        current_line_spans.clear();
    }

    // Handle any unclosed code block
    if in_code_block {
        for code_line in &code_lines {
            let highlighted = highlighter.highlight_line(code_line, code_language);
            lines.push(Line::from(highlighted));
        }
    }

    lines
}

fn parse_header(line: &str) -> Option<Line<'static>> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("# ") {
        Some(Line::from(vec![Span::styled(
            trimmed[2..].to_string(),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )]))
    } else if trimmed.starts_with("## ") {
        Some(Line::from(vec![Span::styled(
            trimmed[3..].to_string(),
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        )]))
    } else if trimmed.starts_with("### ") {
        Some(Line::from(vec![Span::styled(
            trimmed[4..].to_string(),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )]))
    } else {
        None
    }
}

fn parse_list_item(line: &str) -> Option<Line<'static>> {
    let trimmed = line.trim_start();
    let indent = line.len() - trimmed.len();
    let indent_str = " ".repeat(indent);

    if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
        Some(Line::from(vec![
            Span::raw(indent_str),
            Span::styled("• ", Style::default().fg(Color::Yellow)),
            Span::raw(trimmed[2..].to_string()),
        ]))
    } else if let Some(pos) = trimmed.find(". ") {
        if pos > 0 && trimmed[..pos].chars().all(|c| c.is_numeric()) {
            Some(Line::from(vec![
                Span::raw(indent_str),
                Span::styled(
                    format!("{}. ", &trimmed[..pos]),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw(trimmed[pos + 2..].to_string()),
            ]))
        } else {
            None
        }
    } else {
        None
    }
}

fn parse_inline(text: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut chars = text.chars().peekable();
    let mut current = String::new();

    while let Some(ch) = chars.next() {
        match ch {
            '`' => {
                // Code span
                if !current.is_empty() {
                    spans.push(Span::raw(current.clone()));
                    current.clear();
                }

                let mut code = String::new();
                let mut found_closing = false;

                while let Some(ch) = chars.next() {
                    if ch == '`' {
                        found_closing = true;
                        break;
                    }
                    code.push(ch);
                }

                if found_closing {
                    spans.push(Span::styled(
                        code,
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::DIM),
                    ));
                } else {
                    // No closing backtick, treat as regular text
                    current.push('`');
                    current.push_str(&code);
                }
            }
            '*' if chars.peek() == Some(&'*') => {
                // Bold
                chars.next(); // consume second *

                if !current.is_empty() {
                    spans.push(Span::raw(current.clone()));
                    current.clear();
                }

                let mut bold_text = String::new();
                let mut found_closing = false;

                while let Some(ch) = chars.next() {
                    if ch == '*' && chars.peek() == Some(&'*') {
                        chars.next(); // consume second *
                        found_closing = true;
                        break;
                    }
                    bold_text.push(ch);
                }

                if found_closing {
                    spans.push(Span::styled(
                        bold_text,
                        Style::default().add_modifier(Modifier::BOLD),
                    ));
                } else {
                    // No closing **, treat as regular text
                    current.push_str("**");
                    current.push_str(&bold_text);
                }
            }
            '*' | '_' => {
                // Italic
                let marker = ch;

                if !current.is_empty() {
                    spans.push(Span::raw(current.clone()));
                    current.clear();
                }

                let mut italic_text = String::new();
                let mut found_closing = false;

                while let Some(ch) = chars.next() {
                    if ch == marker {
                        found_closing = true;
                        break;
                    }
                    italic_text.push(ch);
                }

                if found_closing && !italic_text.is_empty() {
                    spans.push(Span::styled(
                        italic_text,
                        Style::default().add_modifier(Modifier::ITALIC),
                    ));
                } else {
                    // No closing marker or empty, treat as regular text
                    current.push(marker);
                    current.push_str(&italic_text);
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if !current.is_empty() {
        spans.push(Span::raw(current));
    }

    spans
}
