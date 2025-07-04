use std::time::Duration;

// Spinner frames using Unicode circled operators
// These create a nice rotating animation effect
pub const SPINNER_FRAMES: &[&str] = &[
    "⊕", // U+2295 Circled Plus
    "⊗", // U+2297 Circled Times
    "⊙", // U+2299 Circled Dot Operator
    "⊚", // U+229A Circled Ring Operator
    "⊛", // U+229B Circled Asterisk Operator
    "⊜", // U+229C Circled Equals
    "⊝", // U+229D Circled Dash
    "⊞", // U+229E Squared Plus
    "⊟", // U+229F Squared Minus
    "⊠", // U+22A0 Squared Times
    "⊡", // U+22A1 Squared Dot Operator
];

// Thinking phrases inspired by Gemini CLI
pub const THINKING_PHRASES: &[&str] = &[
    "Thinking deeply...",
    "Pondering your request...",
    "Analyzing the context...",
    "Formulating a response...",
    "Processing information...",
    "Considering possibilities...",
    "Exploring solutions...",
    "Gathering insights...",
    "Synthesizing knowledge...",
    "Crafting a thoughtful reply...",
    "Connecting the dots...",
    "Diving into the details...",
    "Evaluating options...",
    "Building understanding...",
    "Searching memory banks...",
    "Consulting the knowledge graph...",
    "Running semantic analysis...",
    "Activating neural pathways...",
    "Warming up the synapses...",
    "Calibrating response vectors...",
];

#[derive(Debug)]
pub struct SpinnerState {
    pub frame_index: usize,
    pub phrase_index: usize,
    pub elapsed_secs: u64,
    pub start_time: std::time::Instant,
    pub last_frame_time: std::time::Instant,
    pub last_phrase_time: std::time::Instant,
}

impl SpinnerState {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            frame_index: 0,
            phrase_index: 0,
            elapsed_secs: 0,
            start_time: now,
            last_frame_time: now,
            last_phrase_time: now,
        }
    }

    pub fn update(&mut self) {
        let now = std::time::Instant::now();

        // Update spinner frame every 80ms
        if now.duration_since(self.last_frame_time) >= Duration::from_millis(80) {
            self.frame_index = (self.frame_index + 1) % SPINNER_FRAMES.len();
            self.last_frame_time = now;
        }

        // Update thinking phrase every 3 seconds
        if now.duration_since(self.last_phrase_time) >= Duration::from_secs(3) {
            self.phrase_index = (self.phrase_index + 1) % THINKING_PHRASES.len();
            self.last_phrase_time = now;
        }

        // Update elapsed time from start
        self.elapsed_secs = now.duration_since(self.start_time).as_secs();
    }

    pub fn current_frame(&self) -> &'static str {
        SPINNER_FRAMES[self.frame_index]
    }

    pub fn current_phrase(&self) -> &'static str {
        THINKING_PHRASES[self.phrase_index]
    }

    pub fn reset(&mut self) {
        let now = std::time::Instant::now();
        self.frame_index = 0;
        // Simple pseudo-random based on time
        use std::time::{SystemTime, UNIX_EPOCH};
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        self.phrase_index = (millis as usize) % THINKING_PHRASES.len();
        self.elapsed_secs = 0;
        self.start_time = now;
        self.last_frame_time = now;
        self.last_phrase_time = now;
    }
}
