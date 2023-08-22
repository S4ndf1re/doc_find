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

    fn sentences<'a>(&self, input: &'a str) -> Vec<Cow<'a, str>> {
        vec![input.into()]
    }
}
