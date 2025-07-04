use ratatui::{
    style::{Color, Modifier, Style},
    text::Span,
};
use std::collections::HashMap;

/// Simple code syntax highlighter
pub struct CodeHighlighter {
    keywords: HashMap<&'static str, Style>,
}

impl CodeHighlighter {
    pub fn new() -> Self {
        let mut keywords = HashMap::new();

        // Python keywords
        let python_keywords = vec![
            "def", "class", "if", "else", "elif", "for", "while", "return", "import", "from", "as",
            "try", "except", "finally", "with", "async", "await", "yield", "lambda", "pass",
            "break", "continue", "True", "False", "None", "and", "or", "not", "in", "is",
        ];

        for kw in python_keywords {
            keywords.insert(kw, Style::default().fg(Color::Magenta));
        }

        // Common types
        let types = vec![
            "int", "str", "float", "bool", "list", "dict", "set", "tuple",
        ];
        for t in types {
            keywords.insert(t, Style::default().fg(Color::Cyan));
        }

        // Built-in functions
        let builtins = vec!["print", "len", "range", "open", "input", "type"];
        for b in builtins {
            keywords.insert(b, Style::default().fg(Color::Yellow));
        }

        Self { keywords }
    }

    pub fn highlight_line(&self, line: &str, _language: Option<&str>) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        let mut current_word = String::new();
        let mut in_string = false;
        let mut string_char = ' ';
        let mut in_comment = false;

        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            // Handle strings
            if !in_comment && (ch == '"' || ch == '\'') {
                if !in_string {
                    // Start of string
                    if !current_word.is_empty() {
                        spans.push(self.style_word(&current_word));
                        current_word.clear();
                    }
                    in_string = true;
                    string_char = ch;
                    current_word.push(ch);
                } else if ch == string_char {
                    // End of string
                    current_word.push(ch);
                    spans.push(Span::styled(
                        current_word.clone(),
                        Style::default().fg(Color::Green),
                    ));
                    current_word.clear();
                    in_string = false;
                } else {
                    current_word.push(ch);
                }
            } else if in_string {
                current_word.push(ch);
            } else if !in_comment && ch == '#' {
                // Start of comment
                if !current_word.is_empty() {
                    spans.push(self.style_word(&current_word));
                    current_word.clear();
                }
                in_comment = true;
                current_word.push(ch);
            } else if in_comment {
                current_word.push(ch);
            } else if ch.is_alphanumeric() || ch == '_' {
                current_word.push(ch);
            } else {
                // Non-word character
                if !current_word.is_empty() {
                    spans.push(self.style_word(&current_word));
                    current_word.clear();
                }

                // Style operators
                match ch {
                    '+' | '-' | '*' | '/' | '=' | '<' | '>' | '!' | '&' | '|' => {
                        spans.push(Span::styled(
                            ch.to_string(),
                            Style::default().fg(Color::LightRed),
                        ));
                    }
                    '(' | ')' | '[' | ']' | '{' | '}' => {
                        spans.push(Span::styled(
                            ch.to_string(),
                            Style::default().fg(Color::LightBlue),
                        ));
                    }
                    _ => {
                        spans.push(Span::raw(ch.to_string()));
                    }
                }
            }

            i += 1;
        }

        // Handle any remaining content
        if !current_word.is_empty() {
            if in_string {
                spans.push(Span::styled(
                    current_word,
                    Style::default().fg(Color::Green),
                ));
            } else if in_comment {
                spans.push(Span::styled(
                    current_word,
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC),
                ));
            } else {
                spans.push(self.style_word(&current_word));
            }
        }

        spans
    }

    fn style_word(&self, word: &str) -> Span<'static> {
        if let Some(style) = self.keywords.get(word) {
            Span::styled(word.to_string(), *style)
        } else if word.chars().all(|c| c.is_numeric()) {
            // Numbers
            Span::styled(word.to_string(), Style::default().fg(Color::LightMagenta))
        } else {
            Span::raw(word.to_string())
        }
    }
}
