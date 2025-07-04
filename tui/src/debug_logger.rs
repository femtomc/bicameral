use chrono::Local;
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct DebugLogger {
    file: Option<std::fs::File>,
}

static mut DEBUG_LOGGER: Option<Mutex<DebugLogger>> = None;

pub fn init_debug_log() {
    let log_dir = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
        .join(".bicamrl")
        .join("logs");

    // Create directory if it doesn't exist
    let _ = create_dir_all(&log_dir);

    let log_file = log_dir.join(format!(
        "tui_debug_{}.log",
        Local::now().format("%Y%m%d_%H%M%S")
    ));

    if let Ok(file) = OpenOptions::new().create(true).append(true).open(&log_file) {
        unsafe {
            DEBUG_LOGGER = Some(Mutex::new(DebugLogger { file: Some(file) }));
        }
        log_debug(&format!("=== TUI Debug Log Started ==="));
        log_debug(&format!("Log file: {:?}", log_file));
    }
}

pub fn log_debug(msg: &str) {
    unsafe {
        if let Some(ref logger_mutex) = DEBUG_LOGGER {
            if let Ok(mut logger) = logger_mutex.lock() {
                if let Some(ref mut file) = logger.file {
                    let timestamp = Local::now().format("%H:%M:%S%.3f");
                    let _ = writeln!(file, "[{}] {}", timestamp, msg);
                    let _ = file.flush();
                }
            }
        }
    }
}

#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        $crate::debug_logger::log_debug(&format!($($arg)*))
    };
}
