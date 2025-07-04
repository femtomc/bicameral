use unicode_width::UnicodeWidthStr;

/// Calculate the display width of a string, accounting for Unicode characters
pub fn display_width(s: &str) -> usize {
    s.width()
}

/// Truncate a string to fit within a given display width, preserving Unicode boundaries
pub fn truncate_to_width(s: &str, max_width: usize) -> &str {
    if s.width() <= max_width {
        return s;
    }

    let mut width = 0;
    let mut end_byte = 0;

    for (byte_idx, ch) in s.char_indices() {
        let ch_width = ch.to_string().width();
        if width + ch_width > max_width {
            break;
        }
        width += ch_width;
        end_byte = byte_idx + ch.len_utf8();
    }

    &s[..end_byte]
}

/// Split a string into lines that fit within a given display width
pub fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![];
    }

    let mut lines = Vec::new();

    for line in text.lines() {
        if line.width() <= max_width {
            lines.push(line.to_string());
            continue;
        }

        // Need to wrap this line
        let mut current_line = String::new();
        let mut current_width = 0;

        for ch in line.chars() {
            let ch_width = ch.to_string().width();

            if current_width + ch_width > max_width {
                // Start a new line
                lines.push(current_line);
                current_line = String::new();
                current_width = 0;
            }

            current_line.push(ch);
            current_width += ch_width;
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }
    }

    lines
}

/// Check if a character is an emoji or special Unicode character
pub fn is_emoji(ch: char) -> bool {
    matches!(ch as u32,
        0x1F600..=0x1F64F | // Emoticons
        0x1F300..=0x1F5FF | // Misc Symbols and Pictographs
        0x1F680..=0x1F6FF | // Transport and Map
        0x1F700..=0x1F77F | // Alchemical Symbols
        0x1F780..=0x1F7FF | // Geometric Shapes Extended
        0x1F800..=0x1F8FF | // Supplemental Arrows-C
        0x1F900..=0x1F9FF | // Supplemental Symbols and Pictographs
        0x1FA00..=0x1FA6F | // Chess Symbols
        0x1FA70..=0x1FAFF | // Symbols and Pictographs Extended-A
        0x2600..=0x26FF   | // Miscellaneous Symbols
        0x2700..=0x27BF   | // Dingbats
        0x2300..=0x23FF   | // Miscellaneous Technical
        0x2B50..=0x2B55     // Stars
    )
}
