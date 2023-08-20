
use std::borrow::Cow;

pub trait TokenizerStrategie {
    /// 
    fn tokenize<'a>(&self, input: &'a str) -> Vec<Cow<'a, str>>;
    fn sentences<'a>(&self, input: &'a str) -> Vec<Cow<'a, str>>;
}

pub struct SimpleTokenizer;
impl SimpleTokenizer {
    pub fn new() -> Self {
        SimpleTokenizer
    }
}
impl TokenizerStrategie for SimpleTokenizer {
    fn tokenize<'a>(&self, input: &'a str) -> Vec<Cow<'a, str>> {
        input
            .split(|c: char| c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase().into())
            .collect()
    }

    fn sentences<'a>(&self, input: &'a str) -> Vec<Cow<'a, str>> {
        input.split(".").map(|s| s.into()).collect()
    }
}

