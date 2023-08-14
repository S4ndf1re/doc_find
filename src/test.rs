#[test]
fn store_and_find() {
    use crate::{Document, EmptyWordFilter, Index, SimpleTokenizer};

    let mut index = Index::<i64>::new();
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

    index.insert_document(document1);
    index.insert_document(document2);
    index.insert_document(document3);

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
