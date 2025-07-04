use ratatui::style::Color;

// Andromeda color theme (exact colors from the theme)
pub struct AndromedaTheme;

impl AndromedaTheme {
    // Background colors
    pub const BACKGROUND: Color = Color::Rgb(30, 32, 37); // #1e2025
    pub const BACKGROUND_LIGHT: Color = Color::Rgb(43, 47, 56); // #2b2f38

    // Text colors
    pub const FOREGROUND: Color = Color::Rgb(247, 247, 248); // #f7f7f8
    pub const COMMENT: Color = Color::Rgb(175, 171, 178); // #afabb2

    // Accent colors
    pub const TEAL: Color = Color::Rgb(16, 166, 148); // #10a694
    pub const YELLOW: Color = Color::Rgb(254, 229, 107); // #fee56b
    pub const ORANGE: Color = Color::Rgb(242, 156, 19); // #f29c13
    pub const BLUE: Color = Color::Rgb(58, 142, 239); // #3a8eef
    pub const PURPLE: Color = Color::Rgb(199, 77, 237); // #C74DED (keeping this for variety)
    pub const PINK: Color = Color::Rgb(249, 42, 173); // #F92AAD (keeping this)
    pub const GREEN: Color = Color::Rgb(150, 230, 80); // #96E650 (keeping this)
    pub const RED: Color = Color::Rgb(238, 98, 98); // #EE6262 (error color)

    // UI specific colors
    pub const BORDER: Color = Self::BACKGROUND_LIGHT; // #2b2f38
    pub const SELECTION: Color = Color::Rgb(28, 64, 63); // #1C403F

    // Message type colors
    pub const USER_MSG: Color = Self::TEAL;
    pub const ASSISTANT_MSG: Color = Self::YELLOW;
    pub const SYSTEM_MSG: Color = Self::COMMENT;
    pub const INPUT_TEXT: Color = Self::ORANGE;
    pub const HEADER_TEXT: Color = Self::FOREGROUND;
    pub const FOOTER_TEXT: Color = Self::COMMENT;
}
