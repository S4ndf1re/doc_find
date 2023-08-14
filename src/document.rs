use std::rc::Rc;
use crate::TokenizerStrategie;
use crate::WordFilter;

use std::collections::HashMap;

pub trait IntoDocumentString {
    fn into_document_string(self) -> String;
}

impl IntoDocumentString for String {
    fn into_document_string(self) -> String {
        self
    }
}


#[derive(Debug, Clone)]
pub struct Document<I> {
    pub id: Rc<I>,
    pub words: HashMap<String, u64>,
    pub total_words: u64,
}

impl<I> Document<I> {
    pub fn new<D, T, F>(id: I, data: D, filter: &F, tokenizer: &T) -> Self
    where
        D: IntoDocumentString,
        T: TokenizerStrategie,
        F: WordFilter,
    {
        let (words, total) = Self::count_words(data, filter, tokenizer);
        Document {
            id: Rc::new(id),
            words,
            total_words: total,
        }
    }

    fn count_words<D, T, F>(document: D, filter: &F, tokenizer: &T) -> (HashMap<String, u64>, u64)
    where
        D: IntoDocumentString,
        T: TokenizerStrategie,
        F: WordFilter,
    {
        let mut word_count = HashMap::new();
        let content = document.into_document_string();
        let words: Vec<&str> = content
            .split_whitespace()
            .filter(|c| filter.filter(c))
            .collect();
        let content = words.join(" ");

        let mut total = 0;
        for word in tokenizer.tokenize(&content) {
            let count = word_count.entry(word.to_string()).or_insert(0);
            *count += 1;
            total += 1;
        }
        (word_count, total)
    }

    pub fn get_id(&self) -> Rc<I> {
        Rc::clone(&self.id)
    }

    pub fn get_words_ref(&self) -> &HashMap<String, u64> {
        &self.words
    }

    pub fn get_total_count(&self) -> u64 {
        self.total_words
    }

    // Calculate the tf value based on TF(i, j) = log_Lj(1+Freq(i,j)),
    // where i is the word and j the document
    // Lj the total amount of words in the document
    // Freq(i,j) is the frequence of which the word i is contained in j
    //
    // This can be used in combination with TF_IDF ranking.
    // However this should be calculated by the index itself and not the document
    pub fn tf(&self, word: &str) -> f64 {
        let lj = self.total_words as f64;
        let freq = match self.words.get(word) {
            Some(count) => count.clone() as f64,
            None => 0.0,
        };

        f64::log2(1.0 + freq) / f64::log2(lj)
    }
}
