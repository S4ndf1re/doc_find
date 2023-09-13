use anyhow::Error;
use serde::{Deserialize, Serialize};

use crate::util;
use crate::TokenizerStrategie;
use crate::WordFilter;

use std::collections::HashMap;
use std::sync::Arc;

pub trait IntoDocumentString {
    fn into_document_string(self) -> String;
}

impl IntoDocumentString for String {
    fn into_document_string(self) -> String {
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Document<I> {
    pub id: Arc<I>,
    pub words: HashMap<String, u64>,
    pub sentences: Vec<String>,
    pub total_words: u64,
    pub data: String,
}

impl<I> Document<I> 
where I: Send
{
    pub fn new<D, T, F>(id: I, data: D, filter: &F, tokenizer: &T) -> Self
    where
        D: IntoDocumentString,
        T: TokenizerStrategie,
        F: WordFilter,
    {
        let data_str = data.into_document_string();
        let (words, total) = Self::count_words(&data_str, filter, tokenizer);
        let sentences = tokenizer
            .sentences(&data_str)
            .into_iter()
            .map(Into::into)
            .collect();
        Document {
            id: Arc::new(id),
            words,
            sentences,
            total_words: total,
            data: data_str,
        }
    }

    fn count_words<T, F>(document: &str, filter: &F, tokenizer: &T) -> (HashMap<String, u64>, u64)
    where
        T: TokenizerStrategie,
        F: WordFilter,
    {
        let mut word_count = HashMap::new();
        let content = document;
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

    pub fn get_id(&self) -> Arc<I> {
        Arc::clone(&self.id)
    }

    pub fn get_words_ref(&self) -> &HashMap<String, u64> {
        &self.words
    }

    pub fn get_sentences_ref(&self) -> &Vec<String> {
        &self.sentences
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

    pub fn get_embedding(
        &self,
        model: &ort::Session,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<Vec<Vec<f32>>, Error> {
        util::get_embedding(&self.sentences, model, tokenizer)
    }
}

