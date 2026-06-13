use std::collections::HashMap;

/// Typed attribute value stored in a [`Dataset`].
#[derive(Debug, Clone, PartialEq)]
pub enum DatasetValue {
    /// String payload.
    Str(String),
    /// Signed integer payload.
    Int(i64),
    /// Floating-point payload.
    Float(f64),
    /// Boolean payload.
    Bool(bool),
}

impl From<&str> for DatasetValue {
    fn from(value: &str) -> Self {
        Self::Str(value.to_string())
    }
}

impl From<String> for DatasetValue {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl From<i64> for DatasetValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for DatasetValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<bool> for DatasetValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

/// Key-value container for graph metadata and sidecar attributes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Dataset {
    data: HashMap<String, DatasetValue>,
}

impl Dataset {
    /// Creates an empty dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a dataset from key-value pairs.
    pub fn with_pairs<I, K>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, DatasetValue)>,
        K: Into<String>,
    {
        let mut data = HashMap::new();
        for (k, v) in pairs {
            data.insert(k.into(), v);
        }
        Self { data }
    }

    /// Inserts or replaces an attribute.
    pub fn set_attr<K: Into<String>>(&mut self, key: K, value: DatasetValue) {
        self.data.insert(key.into(), value);
    }

    /// Returns an attribute by key.
    pub fn get_attr(&self, key: &str) -> Option<&DatasetValue> {
        self.data.get(key)
    }

    /// Returns an item by key (alias of [`Dataset::get_attr`]).
    pub fn get_item(&self, key: &str) -> Option<&DatasetValue> {
        self.data.get(key)
    }
}

/// Alias for [`Dataset`] matching scikit-network naming.
pub type Bunch = Dataset;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset() {
        let dataset = Dataset::with_pairs([("name", DatasetValue::from("dataset"))]);
        assert_eq!(
            dataset.get_attr("name"),
            Some(&DatasetValue::from("dataset"))
        );
        assert_eq!(
            dataset.get_item("name"),
            Some(&DatasetValue::from("dataset"))
        );
    }
}
