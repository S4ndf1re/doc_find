use anyhow::{anyhow, Error};
use qdrant_client::prelude::{Payload, QdrantClient};
use qdrant_client::qdrant::condition::ConditionOneOf;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::r#match::MatchValue;
use qdrant_client::qdrant::{
    Condition, FieldCondition, Filter, Match, PointStruct, PointsSelector, SearchPoints,
};
use qdrant_client::serde::PayloadConversionError;
use serde::de::DeserializeOwned;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use uuid::Uuid;

use crate::{util, Document, QueryTokenizer, TokenizerStrategie, WordFilter};
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use std::borrow::Cow;
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::Arc;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

const _DOCUMENTS_PREFIX: &str = "~DOCS:";
const _REVERSE_INDEX_PREFIX: &str = "~REV_IDX:";

#[derive(Serialize)]
struct DataIndexStore<'a, I> {
    documents: Vec<(&'a Arc<I>, &'a Document<I>)>,
    reverse_index: Vec<(&'a String, Vec<Arc<I>>)>,
}

#[derive(Deserialize)]
struct DataIndexRead<I> {
    documents: Vec<(I, Document<I>)>,
    reverse_index: Vec<(String, Vec<I>)>,
}

pub struct Index<I>
where
    I: Hash + Eq,
{
    documents: HashMap<Arc<I>, Document<I>>,
    reverse_index: HashMap<String, HashSet<Arc<I>>>,
    tokenizer: tokenizers::Tokenizer,
    model: ort::Session,
    vec_db: QdrantClient,
    collection_name: String,
}

impl<I> Index<I>
where
    I: Hash + Eq + Clone + Serialize + DeserializeOwned + Into<MatchValue> + Display,
{
    pub fn new(
        embed_tokenizer: tokenizers::Tokenizer,
        model: ort::Session,
        client: QdrantClient,
        collection_name: String,
    ) -> Self {
        Index {
            documents: HashMap::new(),
            reverse_index: HashMap::new(),
            tokenizer: embed_tokenizer,
            vec_db: client,
            collection_name,
            model,
        }
    }

    /// This function stores the entire `Index<I>` into a file specified by `path` in json format
    pub async fn store(&self, path: PathBuf) -> Result<(), Error> {
        let mut file_options = OpenOptions::new();
        file_options.create(true).write(true).truncate(true);
        let mut file = file_options.open(path).await?;

        let data = DataIndexStore {
            documents: self.documents.iter().map(|k| k).collect(),
            reverse_index: self
                .reverse_index
                .iter()
                .map(|(k, v)| (k, v.clone().into_iter().map(|v| v).collect()))
                .collect(),
        };

        let buffer = serde_json::to_vec(&data)?;
        file.write(&buffer).await?;

        todo!()
    }

    pub async fn load(
        path: PathBuf,
        embed_tokenizer: tokenizers::Tokenizer,
        model: ort::Session,
        client: QdrantClient,
        collection_name: String,
    ) -> Result<Self, Error> {
        let mut file_options = OpenOptions::new();
        let mut file = file_options
            .read(true)
            .write(false)
            .truncate(false)
            .create(false)
            .open(path)
            .await?;

        let mut buffer = vec![];
        file.read_to_end(&mut buffer).await?;
        let data: DataIndexRead<I> = serde_json::from_slice(&buffer)?;

        let mut documents = HashMap::new();
        data.documents.into_iter().for_each(|(k, v)| {
            documents.insert(Arc::new(k), v);
        });

        let mut reverse_index = HashMap::new();
        data.reverse_index.into_iter().for_each(|(k, v)| {
            let mut set = HashSet::new();
            v.into_iter().for_each(|v| {
                set.insert(Arc::new(v));
            });
            reverse_index.insert(k, set);
        });

        Ok(Self {
            documents,
            reverse_index,
            tokenizer: embed_tokenizer,
            model,
            vec_db: client,
            collection_name,
        })
    }

    pub async fn insert_document<T>(&mut self, doc: Document<I>, tokenizer: &T) -> Result<(), Error>
    where
        T: TokenizerStrategie,
    {
        let id = doc.get_id();
        let words = doc.get_words_ref();
        let embeddings = doc.get_embedding(tokenizer, &self.model, &self.tokenizer)?;

        let payload: Payload = json!( {
            "id": *id
        })
        .try_into()
        .map_err(|e: PayloadConversionError| anyhow!(e))?;

        let points = embeddings
            .into_iter()
            .map(|e| PointStruct::new(Uuid::new_v4().to_string(), e, payload.clone()))
            .collect();

        self.vec_db
            .upsert_points(&self.collection_name, points, None)
            .await?;

        for (word, _) in words {
            if self.reverse_index.contains_key(word) {
                self.reverse_index
                    .get_mut(word)
                    .expect("previous check for existance failed")
                    .insert(id.clone());
            } else {
                let mut set = HashSet::new();
                set.insert(id.clone());
                self.reverse_index.insert(word.to_string(), set);
            }
        }

        self.documents.insert(id.clone(), doc);
        Ok(())
    }

    pub async fn remove_document(&mut self, id: Arc<I>) -> Result<Document<I>, Error> {
        let document = self.documents.remove(id.as_ref());

        let point_selector = PointsSelector {
            points_selector_one_of: Some(PointsSelectorOneOf::Filter(Filter::must(vec![
                Condition {
                    condition_one_of: Some(ConditionOneOf::Field(FieldCondition {
                        key: "id".to_owned(),
                        r#match: Some(Match {
                            match_value: Some(id.as_ref().clone().into()),
                        }),
                        ..Default::default()
                    })),
                },
            ]))),
        };
        self.vec_db
            .delete_points(self.collection_name.clone(), &point_selector, None)
            .await?;

        match document {
            Some(doc) => {
                for (_, list) in &mut self.reverse_index {
                    let id = doc.get_id();
                    list.remove(&id);
                }
                Ok(doc)
            }
            None => Err(anyhow!("no document found for id {}", id)),
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
    ) -> Vec<(f64, &'a Document<I>)>
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

        result.into_iter().map(|(_, v)| v).collect()
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
                let doc = doc.expect("previous check for existance failed");
                let doc_tf_idf = idf * doc.tf(term);
                result.push((doc_tf_idf, doc));
            }
        }
        result
    }

    pub async fn query_embedding<'a, T>(
        &'a self,
        query: &str,
        tokenizer: &T,
    ) -> Result<Vec<(f64, &'a Document<I>)>, Error>
    where
        T: TokenizerStrategie,
    {
        let query_tokenizier = QueryTokenizer::new(tokenizer);
        let mut embeddings =
            util::get_embedding(query, &query_tokenizier, &self.model, &self.tokenizer)?;
        let embedding = embeddings.remove(0);

        let result = self
            .vec_db
            .search_points(&SearchPoints {
                collection_name: self.collection_name.clone(),
                limit: 10,
                with_payload: Some(true.into()),
                vector: embedding,
                ..Default::default()
            })
            .await?;

        let mut similiar_entries = vec![];
        for r in result.result {
            let payload_id: I = serde_json::from_value(r.payload["id"].clone().into_json())?;
            let score = r.score as f64;
            match self.documents.get(&payload_id) {
                Some(doc) => similiar_entries.push((score, doc)),
                None => (),
            };
        }

        Ok(similiar_entries)
    }
}
