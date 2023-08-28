use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    io::{Read, Write},
    path::PathBuf,
    sync::Arc,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use anyhow::{anyhow, Error};
use async_trait::async_trait;
use tokio::{
    fs::OpenOptions,
    io::{AsyncReadExt, AsyncWriteExt},
};

use crate::Document;

/// Defines a storage container that may be used to store data
#[async_trait]
pub trait StorageEngine<I, O> {
    /// Insert a single document into the storage engine.
    /// Also this will update the reverse index.
    async fn insert_document(&mut self, value: Document<I>) -> Result<(), Error>;

    /// Deletes a document from the storage engine.
    /// Also, this will delete the entry from the reverse index.
    async fn remove_document(&mut self, key: Arc<I>) -> Result<Document<I>, Error>;

    /// Get a document from the storage engine.
    async fn get_document<'a>(&'a self, key: Arc<I>) -> Result<&'a Document<I>, Error>;

    async fn get_document_len(&self) -> usize;

    /// Get all documents from the storage engine that containt the given token.
    async fn get_reverse_documents<'a>(
        &'a self,
        token: &str,
    ) -> Result<Vec<&'a Document<I>>, Error>;

    /// Save the storage engine to disk (or any other media implemented by the StorageEngine).
    async fn save(&self, _options: O) -> Result<(), Error>
    where
        O: Send + Sync + 'static,
    {
        Ok(())
    }

    /// Load the storage engine from disk (or any other media implemented by the StorageEngine).
    async fn load(&mut self, _options: O) -> Result<(), Error>
    where
        O: Send + Sync + 'static,
    {
        Ok(())
    }
}

pub struct MemoryStorage<I>
where
    I: Hash + Eq + Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    documents: HashMap<Arc<I>, Document<I>>,
    reverse_index: HashMap<String, HashSet<Arc<I>>>,
    path: PathBuf,
}

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

impl<I> MemoryStorage<I>
where
    I: Hash + Eq + Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    pub fn new<O>(path: O) -> Self
    where
        O: Into<PathBuf>,
    {
        let mut s = Self {
            documents: HashMap::new(),
            reverse_index: HashMap::new(),
            path: path.into(),
        };

        // NOTE: this error can be ignored, because the resulting structure will be initial
        let _ = s.load_sync(s.path.clone());

        s
    }
}

#[async_trait]
impl<I, O> StorageEngine<I, O> for MemoryStorage<I>
where
    I: Hash + Eq + Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    O: Send + Sync + Into<PathBuf> + 'static,
{
    async fn insert_document(&mut self, document: Document<I>) -> Result<(), Error> {
        let words = document.get_words_ref();
        let key = document.get_id();

        for (word, _) in words {
            if self.reverse_index.contains_key(word) {
                self.reverse_index
                    .get_mut(word)
                    .expect("already checked")
                    .insert(key.clone());
            } else {
                let entry = self
                    .reverse_index
                    .entry(word.clone())
                    .or_insert(HashSet::new());
                entry.insert(key.clone());
            }
        }

        self.documents.insert(key, document);
        Ok(())
    }

    async fn remove_document(&mut self, key: Arc<I>) -> Result<Document<I>, Error> {
        let document = self.documents.remove(&key);

        if let Some(doc) = &document {
            let words = doc.get_words_ref();

            for (word, _) in words {
                if let Some(entry) = self.reverse_index.get_mut(word) {
                    entry.remove(&key);
                }
            }
            Ok(document.unwrap())
        } else {
            Err(anyhow!("document not found"))
        }
    }

    async fn get_document<'a>(&'a self, key: Arc<I>) -> Result<&'a Document<I>, Error> {
        self.documents
            .get(&key)
            .ok_or(anyhow!("document not found"))
    }

    async fn get_document_len(&self) -> usize {
        self.documents.len()
    }

    async fn get_reverse_documents<'a>(
        &'a self,
        token: &str,
    ) -> Result<Vec<&'a Document<I>>, Error> {
        if let Some(entry) = self.reverse_index.get(token) {
            Ok(entry
                .iter()
                .filter_map(|key| self.documents.get(key))
                .collect())
        } else {
            Err(anyhow!("token not found"))
        }
    }

    async fn save(&self, options: O) -> Result<(), Error> {
        let path: PathBuf = options.into();

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

        Ok(())
    }

    async fn load(&mut self, options: O) -> Result<(), Error> {
        let path: PathBuf = options.into();

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

        self.documents = documents;
        self.reverse_index = reverse_index;
        Ok(())
    }
}

impl<T> MemoryStorage<T>
where
    T: Hash + Eq + Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    fn load_sync<O>(&mut self, options: O) -> Result<(), Error>
    where
        O: Into<PathBuf>,
    {
        let path = options.into();

        let mut file_options = std::fs::OpenOptions::new();
        let mut file = file_options
            .read(true)
            .write(false)
            .truncate(false)
            .create(false)
            .open(path)?;

        let mut buffer = vec![];
        file.read_to_end(&mut buffer)?;
        let data: DataIndexRead<T> = serde_json::from_slice(&buffer)?;

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

        self.documents = documents;
        self.reverse_index = reverse_index;

        Ok(())
    }

    pub fn save_sync<O>(&self, options: O) -> Result<(), Error>
    where
        O: Into<PathBuf>,
    {
        let path: PathBuf = options.into();

        let mut file_options = std::fs::OpenOptions::new();
        file_options.create(true).write(true).truncate(true);
        let mut file = file_options.open(path)?;

        let data = DataIndexStore {
            documents: self.documents.iter().map(|k| k).collect(),
            reverse_index: self
                .reverse_index
                .iter()
                .map(|(k, v)| (k, v.clone().into_iter().map(|v| v).collect()))
                .collect(),
        };

        let buffer = serde_json::to_vec(&data)?;
        file.write(&buffer)?;

        Ok(())
    }
}

impl<T> Drop for MemoryStorage<T>
where
    T: Hash + Eq + Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    fn drop(&mut self) {
        let _ = self.save_sync(self.path.clone());
    }
}
