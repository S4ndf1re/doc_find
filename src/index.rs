use std::rc::Rc;
use crate::Document;
use std::{collections::HashMap, hash::Hash};

pub struct Index<I>
where
    I: Hash + Eq + Clone,
{
    documents: HashMap<Rc<I>, Document<I>>,
    reverse_index: HashMap<String, Vec<Rc<I>>>,
}

impl<I> Index<I>
where
    I: Hash + Eq + Clone,
{
    pub fn new() -> Self {
        Index {
            documents: HashMap::new(),
            reverse_index: HashMap::new(),
        }
    }

    pub fn insert_document(&mut self, doc: Document<I>) {

        let id = doc.id.clone();
        let words = doc.get_words_ref();

        for (word, _) in words {
            if self.reverse_index.contains_key(word) {
                self.reverse_index.get_mut(word).unwrap().push(id.clone());
            } else {
                self.reverse_index.insert(word.to_string(), vec![id.clone()]);
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
                    let idx = list.iter().position(|x| *x == id);
                    if idx.is_some() {
                        list.swap_remove(idx.unwrap());
                    }
                }
                None
            }
            None => None,
        }
    }
}
