use std::collections::HashMap;

use crate::Document;

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum ResultType {
    TfIdf,
    Vector,
    Bm25,
}

pub struct QueryResult<'a, I> {
    results: HashMap<ResultType, Vec<(f64, &'a Document<I>)>>,
}

impl<'a, I> QueryResult<'a, I> {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    pub fn add(&mut self, result_type: ResultType, score: f64, document: &'a Document<I>) {
        self.results
            .entry(result_type)
            .or_insert_with(Vec::new)
            .push((score, document));
    }

    pub fn add_vec(&mut self, result_type: ResultType, list: &Vec<(f64, &'a Document<I>)>) {
        let entry = self.results.entry(result_type).or_insert_with(Vec::new);
        entry.extend(list)
    }

    fn max_for_type(&self, r_type: &ResultType) -> f64 {
        let mut max_value = 0.0;
        if let Some(list) = self.results.get(r_type) {
            for (v, _) in list {
                if v > &max_value {
                    max_value = v.clone();
                }
            }
        }
        max_value
    }

    fn normalize(&mut self) {
        let max_values: Vec<(ResultType, f64)> = self
            .results
            .iter()
            .map(|(k, _)| (k.clone(), self.max_for_type(k)))
            .collect();

        for (key, max) in max_values {
            let values = self.results.get_mut(&key);
            if let Some(list) = values {
                for value in list {
                    value.0 = value.0 / max;
                }
            }
        }
    }

    pub fn collect(mut self) -> Vec<(f64, &'a Document<I>)> {
        self.normalize();

        let mut result: Vec<_> = self.results.into_iter().flat_map(|(_, v)| v).collect();
        result.sort_by(|a, b| a.0.total_cmp(&b.0));
        result
    }
}
