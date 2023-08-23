use std::{collections::HashSet, fs, io, path::PathBuf};

/// Filter definition to filter out useless or bad words
pub trait WordFilter {
    /// filter a token
    fn filter(&self, input: &str) -> bool;
}

pub struct SimpleWordFilter {
    forbidden_words: HashSet<String>,
}

impl SimpleWordFilter {
    pub fn new(path: PathBuf) -> Result<Self, io::Error> {
        let words_file = fs::read_to_string(path)?;
        let forbidden_words = words_file
            .lines()
            .map(|line| line.trim().to_lowercase())
            .collect::<HashSet<String>>();

        Ok(SimpleWordFilter { forbidden_words })
    }
}

impl WordFilter for SimpleWordFilter {
    fn filter(&self, input: &str) -> bool {
        !self.forbidden_words.contains(input)
    }
}

pub struct EmptyWordFilter {}

impl WordFilter for EmptyWordFilter {
    fn filter(&self, _input: &str) -> bool {
        true
    }
}
