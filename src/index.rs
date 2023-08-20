use crate::{Document, TokenizerStrategie, WordFilter};
use std::borrow::Cow;
use std::rc::Rc;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

pub struct Index<I>
where
    I: Hash + Eq + Clone,
{
    documents: HashMap<Rc<I>, Document<I>>,
    reverse_index: HashMap<String, HashSet<Rc<I>>>,
    tokenizer: tokenizers::Tokenizer,
    model : ort::Session,

}

impl<I> Index<I>
where
    I: Hash + Eq + Clone,
{
    pub fn new(embed_tokenizer: tokenizers::Tokenizer, model: ort::Session) -> Self {
        Index {
            documents: HashMap::new(),
            reverse_index: HashMap::new(),
            tokenizer: embed_tokenizer,
            model,
        }
    }

    pub fn insert_document<T>(&mut self, doc: Document<I>, tokenizer: &T) 
    where T: TokenizerStrategie {
        let id = doc.id.clone();
        let words = doc.get_words_ref();
        let _embeddings = doc.sentences_to_vec(tokenizer, &self.model, &self.tokenizer);

        for (word, _) in words {
            if self.reverse_index.contains_key(word) {
                self.reverse_index.get_mut(word).unwrap().insert(id.clone());
            } else {
                let mut set = HashSet::new();
                set.insert(id.clone());
                self.reverse_index.insert(word.to_string(), set);
            }
        }

        self.documents.insert(id.clone(), doc);
    }

    pub fn remove_document(&mut self, id: &I) -> Option<Document<I>> {
        let document = self.documents.remove(id);
        match document {
            Some(doc) => {
                for (_, list) in &mut self.reverse_index {
                    let id = doc.get_id();
                    list.remove(&id);
                }
                None
            }
            None => None,
        }
    }

    /// calculate the inverse document frequeny idf(t)=log N/J,
    /// where N is the total number of document
    /// and J is the number of documents that contain the term t
    fn idf(&self, term: &str) -> f64 {
        let n = self.documents.len() as f64;
        let j = match self.reverse_index.get(term) {
            Some(l) => l.len() as f64,
            None => 0.0,
        };

        f64::log10(n / j)
    }

    // calculate tf_idf for all documents. 
    // this can take a query string, that can contain multiple tokens (using the same tokenizer as
    // the documents
    pub fn tf_idf_all<'a, T, F>(
        &'a self,
        query: &str,
        tokenizer: &T,
        filter: &F,
    ) -> HashMap<Rc<I>, (f64, &'a Document<I>)>
    where
        T: TokenizerStrategie,
        F: WordFilter,
    {
        let mut result = HashMap::new();
        let tokens: Vec<Cow<'_, str>> = tokenizer
            .tokenize(query)
            .into_iter()
            .filter(|f| filter.filter(f))
            .collect();

        for token in &tokens {
            let documents = self.tf_idf(token);

            for (score, doc) in documents {
                let id = doc.get_id();
                let entry = result.entry(id).or_insert((0.0, doc));
                entry.0 += score;
            }
        }

        result
    }

    /// Calculate tf_idf of all documents that contain the term `term`
    pub fn tf_idf<'a>(&'a self, term: &str) -> Vec<(f64, &'a Document<I>)> {
        let mut result = vec![];
        let idf = self.idf(term);

        let empty_set = HashSet::new();
        let relevant_docs = match self.reverse_index.get(term) {
            Some(l) => l,
            None => &empty_set,
        };

        for doc_id in relevant_docs {
            let doc = self.documents.get(doc_id);
            if doc.is_some() {
                let doc = doc.unwrap();
                let doc_tf_idf = idf * doc.tf(term);
                result.push((doc_tf_idf, doc));
            }
        }
        result
    }
}
