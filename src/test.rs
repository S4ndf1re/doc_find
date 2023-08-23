use qdrant_client::{
    prelude::QdrantClient,
    qdrant::{vectors_config::Config, CreateCollection, Distance, VectorParams, VectorsConfig},
};

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

    let mut index = Index::<i64>::new(onnx_tokenizer, model, client, collection_name);
    let tokenizer = SimpleTokenizer::new();
    let filter = EmptyWordFilter {};

    let document1 = Document::new(
        1,
        "a quick brown fox lazily shakes a banana tree".to_string(),
        &filter,
        &tokenizer,
    );
    let document2 = Document::new(
        2,
        "a quick brown fox pulls a gun fast".to_string(),
        &filter,
        &tokenizer,
    );
    let document3 = Document::new(3, "test me fast".to_string(), &filter, &tokenizer);

    runtime.block_on(async {
        index.insert_document(document1, &tokenizer).await.unwrap();
        index.insert_document(document2, &tokenizer).await.unwrap();
        index.insert_document(document3, &tokenizer).await.unwrap();
    });

    let result = index.tf_idf_all("brown fox", &tokenizer, &filter);
    assert!(result.len() == 2);
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &1));
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &2));
    assert!(!result.iter().any(|(_, e)| e.id.as_ref() == &3));

    let result = index.tf_idf_all("fast", &tokenizer, &filter);
    assert!(result.len() == 2);
    assert!(!result.iter().any(|(_, e)| e.id.as_ref() == &1));
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &2));
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &3));
}
