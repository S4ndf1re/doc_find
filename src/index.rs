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
use uuid::Uuid;

use crate::{util, Document, QueryTokenizer, StorageEngine, TokenizerStrategie, WordFilter};
use serde::Serialize;
use serde_json::json;
use std::borrow::Cow;
use std::fmt::Display;
use std::sync::Arc;
use std::{collections::HashMap, hash::Hash};

const _DOCUMENTS_PREFIX: &str = "~DOCS:";
const _REVERSE_INDEX_PREFIX: &str = "~REV_IDX:";

pub struct QdrantOptions {
    client: QdrantClient,
    collection_name: String,
}

impl QdrantOptions {
    pub fn new(client: QdrantClient, collection_name: String) -> Self {
        Self {
            client,
            collection_name,
        }
    }
}

pub struct Index<I, ST, O> {
    tokenizer: tokenizers::Tokenizer,
    model: ort::InMemorySession<'static>,
    vec_db: Option<QdrantOptions>,
    storage: ST,
    _phantom_i: std::marker::PhantomData<I>,
    _phantom_o: std::marker::PhantomData<O>,
}

impl<I, ST, O> Index<I, ST, O>
where
    I: Hash + Eq + Clone + Serialize + DeserializeOwned + Into<MatchValue> + Display,
    ST: StorageEngine<I, O>,
{
    /// Create a new `Index<I>` that can store multiple `Documents<I>` and query over its data.
    pub fn new(client: Option<QdrantOptions>, storage: ST) -> Self {
        let model_bytes = include_bytes!("../model/pytorch_model.onnx.large");
        let tokens_bytes = include_bytes!("../model/tokens.json.large");

        let environment = ort::Environment::builder()
            .with_name("Hugging Face Embedding")
            .with_execution_providers([ort::ExecutionProvider::CUDA(Default::default())])
            .build()
            .unwrap()
            .into_arc();

        let model = ort::SessionBuilder::new(&environment)
            .unwrap()
            .with_optimization_level(ort::GraphOptimizationLevel::Level1)
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .with_model_from_memory(model_bytes)
            .unwrap();

        let onnx_tokenizer = tokenizers::Tokenizer::from_bytes(tokens_bytes).unwrap();

        Index {
            tokenizer: onnx_tokenizer,
            vec_db: client,
            model,
            storage,
            _phantom_i: std::marker::PhantomData,
            _phantom_o: std::marker::PhantomData,
        }
    }

    /// Insert a single `Document<I>` into the `Index<I>` using a custom Tokenizer.
    /// The Tokenizer should be the same as used for the `Document<I>`s creation.
    pub async fn insert_document(&mut self, doc: Document<I>) -> Result<(), Error> {
        let id = doc.get_id();

        if self.vec_db.is_some() {
            let QdrantOptions {
                client,
                collection_name,
            } = self
                .vec_db
                .as_ref()
                .expect("already checked before. This should be Some");

            let embeddings = doc.get_embedding(&self.model, &self.tokenizer)?;

            let payload: Payload = json!( {
                "id": *id
            })
            .try_into()
            .map_err(|e: PayloadConversionError| anyhow!(e))?;

            let points = embeddings
                .into_iter()
                .map(|e| PointStruct::new(Uuid::new_v4().to_string(), e, payload.clone()))
                .collect();

            client.upsert_points(collection_name, points, None).await?;
        }

        self.storage.insert_document(doc).await
    }

    /// Remove a single `Document<I>` and return it.
    /// When the document is not found, an error is returned.
    /// Also, when the request to qdrant failed, an error is returned, too.
    pub async fn remove_document(&mut self, id: Arc<I>) -> Result<Document<I>, Error> {
        if self.vec_db.is_some() {
            let QdrantOptions {
                client,
                collection_name,
            } = self
                .vec_db
                .as_ref()
                .expect("already checked before. This should be Some");

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
            client
                .delete_points(collection_name, &point_selector, None)
                .await?;
        }
        self.storage.remove_document(id).await
    }

    /// calculate the inverse document frequeny idf(t)=log N/J,
    /// where N is the total number of document
    /// and J is the number of documents that contain the term t
    async fn idf(&self, term: &str) -> f64 {
        let n = self.storage.get_document_len().await as f64;
        let j = match self.storage.get_reverse_documents(term).await {
            Ok(l) => l.len() as f64,
            Err(_) => 0.0,
        };

        f64::log10(n / j)
    }

    /// calculate tf_idf for all documents.
    /// this can take a query string, that can contain multiple tokens (using the same tokenizer as
    /// the documents
    pub async fn tf_idf_all<'a, T, F>(
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
            let documents = match self.tf_idf(token).await {
                Ok(docs) => docs,
                Err(_) => Vec::new(),
            };

            for (score, doc) in documents {
                let id = doc.get_id();
                let entry = result.entry(id).or_insert((0.0, doc));
                entry.0 += score;
            }
        }

        result.into_iter().map(|(_, v)| v).collect()
    }

    /// Calculate tf_idf of all documents that contain the term `term`
    pub async fn tf_idf<'a>(&'a self, term: &str) -> Result<Vec<(f64, &'a Document<I>)>, Error> {
        let mut result = vec![];
        let idf = self.idf(term).await;

        let relevant_docs = match self.storage.get_reverse_documents(term).await {
            Ok(docs) => docs,
            Err(_) => Vec::new(),
        };

        for doc_id in relevant_docs {
            let doc = self.storage.get_document(doc_id.get_id()).await?;
            let doc_tf_idf = idf * doc.tf(term);
            result.push((doc_tf_idf, doc));
        }
        Ok(result)
    }

    /// Query `Document<I>`s using the embeddings generated during `Self::insert_document`.
    /// This will not run `tf_idf` or any other serach.
    /// At the end, a list of all found `Document<I>`s will get returned, tupled with the vector
    /// distance to the query.
    pub async fn query_embedding<'a, T>(
        &'a self,
        query: &str,
        tokenizer: &T,
    ) -> Result<Vec<(f64, &'a Document<I>)>, Error>
    where
        T: TokenizerStrategie,
    {
        if self.vec_db.is_none() {
            return Err(anyhow!("no qdrant client options set"));
        }

        let QdrantOptions {
            client,
            collection_name,
        } = self
            .vec_db
            .as_ref()
            .expect("already checked before. This should be Some");

        let query_tokenizier = QueryTokenizer::new(tokenizer);
        let sentences = query_tokenizier.sentences(query);

        let mut embeddings = util::get_embedding(&sentences, &self.model, &self.tokenizer)?;
        let embedding = embeddings.remove(0);

        let result = client
            .search_points(&SearchPoints {
                collection_name: collection_name.clone(),
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
            match self.storage.get_document(Arc::new(payload_id)).await {
                Ok(doc) => similiar_entries.push((score, doc)),
                Err(_) => (),
            };
        }

        Ok(similiar_entries)
    }
}
