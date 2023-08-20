use ndarray::Array;
use ndarray::Array1;
use ndarray::Axis;
use ndarray::CowArray;
use ort::Value;
use ort::tensor::OrtOwnedTensor;

use crate::TokenizerStrategie;
use crate::WordFilter;
use std::rc::Rc;

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
    pub data: String,
}

impl<I> Document<I> {
    pub fn new<D, T, F>(id: I, data: D, filter: &F, tokenizer: &T) -> Self
    where
        D: IntoDocumentString,
        T: TokenizerStrategie,
        F: WordFilter,
    {
        let data_str = data.into_document_string();
        let (words, total) = Self::count_words(&data_str, filter, tokenizer);
        Document {
            id: Rc::new(id),
            words,
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

    pub fn sentences_to_vec<T>(
        &self,
        data_tokenizer: &T,
        model: &ort::Session,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<Vec<Vec<f32>>, tokenizers::Error>
    where
        T: TokenizerStrategie,
    {
        let mut result = vec![];
        let sentences = data_tokenizer.sentences(&self.data);
        for sentence in &sentences {
            let tokens = tokenizer.encode(sentence.as_ref(), false)?;
            let ids = tokens.get_ids();
            let shape = (1, ids.len());
            let ids = CowArray::from(Array1::from_iter(ids.into_iter().map(|s| *s as i64)).into_dyn());
            // let attentions = CowArray::from(Array::from_elem(shape, 1_i64).into_dyn());
            let type_ids = CowArray::from(Array::from_elem(shape, 0_i64).into_dyn());

            let embedding_result = model.run(vec![
                Value::from_array(model.allocator(), &ids)?,
                // Value::from_array(model.allocator(), &attentions)?,
                Value::from_array(model.allocator(), &type_ids)?,
            ])?;
            let output: OrtOwnedTensor<f32, _> = embedding_result[0].try_extract()?;
            let pooled = output.view().mean_axis(Axis(1)).ok_or(tokenizers::Error::from("pooling failed"))?;
            let embedding = pooled.as_slice().ok_or(tokenizers::Error::from("can't retreive pooling as slice"))?.to_vec();
            result.push(embedding);
        }

        Ok(result)
    }
}
