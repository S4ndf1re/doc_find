use std::borrow::Cow;

/// Defines a custom Tokenizer Strategie that is used for `Document<I>` token generation
pub trait TokenizerStrategie {
    /// generate a list of single tokens
    fn tokenize<'a>(&self, input: &'a str) -> Vec<Cow<'a, str>>;
    /// generate a list of sentences
    fn sentences<'a>(&self, input: &'a str) -> Vec<String>;
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

    fn sentences<'a>(&self, input: &'a str) -> Vec<String> {
        input.split(".\n").map(|s| s.to_owned()).collect()
    }
}

pub struct QueryTokenizer<'b, T>
where
    T: TokenizerStrategie,
{
    normal: &'b T,
}

impl<'b, T> QueryTokenizer<'b, T>
where
    T: TokenizerStrategie,
{
    pub fn new(other: &'b T) -> Self {
        QueryTokenizer { normal: other }
    }
}

impl<'b, T> TokenizerStrategie for QueryTokenizer<'b, T>
where
    T: TokenizerStrategie,
{
    fn tokenize<'a>(&self, input: &'a str) -> Vec<Cow<'a, str>> {
        self.normal.tokenize(input)
    }

    fn sentences<'a>(&self, input: &'a str) -> Vec<String> {
        vec![input.to_owned()]
    }
}
