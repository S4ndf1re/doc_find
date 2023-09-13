use std::path::PathBuf;

use qdrant_client::{
    prelude::QdrantClient,
    qdrant::{vectors_config::Config, CreateCollection, Distance, VectorParams, VectorsConfig},
};

use crate::{MemoryStorage, OptionType, QdrantOptions, QueryOption, EMBEDDING_DIM};

#[test]
fn store_and_find() {
    use crate::{Document, EmptyWordFilter, Index, SimpleTokenizer};

    let timer = std::time::Instant::now();

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
                    size: EMBEDDING_DIM,
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        }))
        .unwrap();

    let opts = QdrantOptions::new(client, collection_name);
    let storage = MemoryStorage::new("index.json");

    let mut index = Index::<i64, _, PathBuf>::new(Some(opts), storage);
    let tokenizer = SimpleTokenizer::new();
    let filter = EmptyWordFilter {};

    println!("Initialization took {} ms", timer.elapsed().as_millis());

    let timer = std::time::Instant::now();
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
        index.insert_document(document1).await.unwrap();
        index.insert_document(document2).await.unwrap();
        index.insert_document(document3).await.unwrap();
    });

    println!("Insertion took {} ms", timer.elapsed().as_millis());

    let options = QueryOption::new().add(OptionType::TfIdf).build();

    let timer = std::time::Instant::now();

    let result = runtime.block_on(async {
        index
            .query("brown fox", &tokenizer, &filter, Some(options.clone()))
            .await
    });

    let result = result.unwrap();
    let result = result.collect();
    assert!(result.len() == 2);
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &1));
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &2));
    assert!(!result.iter().any(|(_, e)| e.id.as_ref() == &3));

    let result = runtime.block_on(async {
        index
            .query("fast", &tokenizer, &filter, Some(options.clone()))
            .await
    });

    let result = result.unwrap();
    let result = result.collect();
    assert!(result.len() == 2);
    assert!(!result.iter().any(|(_, e)| e.id.as_ref() == &1));
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &2));
    assert!(result.iter().any(|(_, e)| e.id.as_ref() == &3));

    println!("Query took {} ms", timer.elapsed().as_millis());
}
