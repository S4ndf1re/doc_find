use std::fmt::Display;

use qdrant_client::{
    prelude::QdrantClient,
    qdrant::{
        r#match::MatchValue, vectors_config::Config, CreateCollection, Distance, VectorParams,
        VectorsConfig,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Hash, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct I64(i64);

impl Into<tikv_client::Key> for I64 {
    fn into(self) -> tikv_client::Key {
        serde_json::to_vec(&self.0).unwrap().into()
    }
}

impl From<tikv_client::Key> for I64 {
    fn from(value: tikv_client::Key) -> Self {
        let v: Vec<u8> = value.into();
        Self(serde_json::from_slice(&v).unwrap())
    }
}

impl Into<tikv_client::Value> for I64 {
    fn into(self) -> tikv_client::Value {
        serde_json::to_vec(&self.0).unwrap().into()
    }
}

impl Into<MatchValue> for I64 {
    fn into(self) -> MatchValue {
        MatchValue::Integer(self.0)
    }
}

impl Display for I64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[test]
fn store_and_find() {
    use crate::{Document, EmptyWordFilter, Index, SimpleTokenizer};

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
        .with_model_from_file("model/pytorch_model.onnx")
        .unwrap();

    let onnx_tokenizer = tokenizers::Tokenizer::from_file("model/tokens.json").unwrap();

    let client = QdrantClient::from_url("http://localhost:6334")
        .build()
        .unwrap();
    let collection_name = "test".to_owned();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime
        .block_on(client.delete_collection(&collection_name))
        .unwrap();
    runtime
        .block_on(client.create_collection(&CreateCollection {
            collection_name: collection_name.clone(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 768,
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        }))
        .unwrap();

    let mut index = Index::<I64>::new(onnx_tokenizer, model, client, collection_name);
    let tokenizer = SimpleTokenizer::new();
    let filter = EmptyWordFilter {};

    let document1 = Document::new(
        I64(1),
        "a quick brown fox lazily shakes a banana tree".to_string(),
        &filter,
        &tokenizer,
    );
    let document2 = Document::new(
        I64(2),
        "a quick brown fox pulls a gun fast".to_string(),
        &filter,
        &tokenizer,
    );
    let document3 = Document::new(I64(3), "test me fast".to_string(), &filter, &tokenizer);

    runtime.block_on(async {
        index.insert_document(document1).await.unwrap();
        index.insert_document(document2).await.unwrap();
        index.insert_document(document3).await.unwrap();
    });

    let result = index.tf_idf_all("brown fox", &tokenizer, &filter);
    assert!(result.len() == 2);
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &I64(1)));
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &I64(2)));
    assert!(!result.iter().any(|(_, e)| e.id.as_ref() == &I64(3)));

    let result = index.tf_idf_all("fast", &tokenizer, &filter);
    assert!(result.len() == 2);
    assert!(!result.iter().any(|(_, e)| e.id.as_ref() == &I64(1)));
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &I64(2)));
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &I64(3)));
}
