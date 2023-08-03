
pub trait TokenizerStrategie {
    fn tokenize(&self, input: &str) -> Vec<String>;
}

pub struct SimpleTokenizer;
impl SimpleTokenizer {
    pub fn new() -> Self {
        SimpleTokenizer
    }
}
impl TokenizerStrategie for SimpleTokenizer {
    fn tokenize(&self, input: &str) -> Vec<String> {
        input
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }
}

