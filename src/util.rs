use anyhow::{anyhow, Error};
use ndarray::{Array, Axis, CowArray};
use ort::{tensor::OrtOwnedTensor, Value};

/// Generate sentence embeddings for all `sentences`.
/// Sentences are generated using a sentece embedding model and the onnx runtime.
/// The `tokenizer` is a huggingface implementation of their tokenizer specification.
/// It defines all valid words, as well as vector length and empty vector slot values.
pub fn get_embedding(
    sentences: &Vec<String>,
    model: &ort::Session,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<Vec<f32>>, Error> {
    let mut result = vec![];
    for sentence in sentences {
        let tokens = tokenizer
            .encode(sentence.as_str(), false)
            .map_err(|e| anyhow!(e))?;
        let ids = tokens.get_ids();
        let shape = (1, ids.len());
        let mut temp = Array::from_elem(shape, 0_i64);
        for (i, id) in ids.into_iter().enumerate() {
            temp[(0, i)] = *id as i64;
        }

        let ids = CowArray::from(temp.into_dyn());
        let attentions = CowArray::from(Array::from_elem(shape, 1_i64).into_dyn());

        let type_ids = CowArray::from(Array::from_elem(shape, 0_i64).into_dyn());

        let embedding_result = model.run(vec![
            Value::from_array(model.allocator(), &ids)?,
            Value::from_array(model.allocator(), &attentions)?,
            Value::from_array(model.allocator(), &type_ids)?,
        ])?;
        let output: OrtOwnedTensor<f32, _> = embedding_result[0].try_extract()?;
        let pooled = output
            .view()
            .mean_axis(Axis(1))
            .ok_or(Error::msg("pooling failed"))?;
        let embedding = pooled
            .as_slice()
            .ok_or(Error::msg("can't retreive pooling as slice"))?
            .to_vec();
        result.push(embedding);
    }

    Ok(result)
}
