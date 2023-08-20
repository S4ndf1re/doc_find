#[test]
fn store_and_find() {
    use crate::{Document, EmptyWordFilter, Index, SimpleTokenizer};

    let environment = ort::Environment::builder()
        .with_name("Hugging Face Embedding")
        .with_execution_providers([ort::ExecutionProvider::CUDA(Default::default())])
        .build().unwrap()
        .into_arc();

    let model = ort::SessionBuilder::new(&environment).unwrap()
        .with_optimization_level(ort::GraphOptimizationLevel::Level1).unwrap()
        .with_intra_threads(1).unwrap()
        .with_model_from_file("model/pytorch_model.onnx").unwrap();

    let onnx_tokenizer = tokenizers::Tokenizer::from_file("model/tokens.json").unwrap();

    let mut index = Index::<i64>::new(onnx_tokenizer, model);
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

    index.insert_document(document1, &tokenizer);
    index.insert_document(document2, &tokenizer);
    index.insert_document(document3, &tokenizer);

    let result = index.tf_idf_all("brown fox", &tokenizer, &filter);
    assert!(result.len() == 2);
    assert!(result.contains_key(&1));
    assert!(result.contains_key(&2));
    assert!(!result.contains_key(&3));

    let result = index.tf_idf_all("fast", &tokenizer, &filter);
    assert!(result.len() == 2);
    assert!(!result.contains_key(&1));
    assert!(result.contains_key(&2));
    assert!(result.contains_key(&3));
}
