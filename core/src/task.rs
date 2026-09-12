use std::fmt;

#[derive(Debug)]
pub struct TaskCancelled;

impl fmt::Display for TaskCancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("audio processing cancelled")
    }
}

impl std::error::Error for TaskCancelled {}
