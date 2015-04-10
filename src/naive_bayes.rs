use std::collections::{HashMap, HashSet};
use std::f64;
use regex::Regex;
use rustc_serialize::json;

static DEFAULT_SMOOTHING: f64 = 1.0f64;

/// Naive Bayes classifier
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct Classifier {
    vocab: HashSet<String>,
    num_examples: u32,
    smoothing: f64,
    classifications: HashMap<String, Classification>
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
struct Classification {
    label: String,
    num_examples: u32,
    num_words: u32,
    probability: f64,
    default_word_probability: f64,
    words: HashMap<String, (u32, f64)>,
}

impl Classifier {
    
    /// Creates a new classifier
    pub fn new() -> Classifier {
        Classifier {
            vocab: HashSet::new(),
            num_examples: 0u32,
            smoothing: DEFAULT_SMOOTHING,
            classifications: HashMap::new(),
        }
    }

    /// Takes a document that has been tokenized into a vector of strings
    /// and a label and adds the document to the list of documents that the
    /// classifier is aware of and will train on next time the `train()` method is called
    pub fn add_document_tokenized(&mut self, document: &Vec<String>, label: &String) {
        if document.len() == 0 { return; }
        
        // make sure the classification already exists
        if !self.classifications.contains_key(label) {
            let c = Classification::new(label);
            self.classifications.insert(label.clone(), c);
        }

        let mut classification = self.classifications.get_mut(label).unwrap();
                
        for word in document.iter() {
            classification.add_word(word);
            self.vocab.insert(word.to_string());
        }

        self.num_examples += 1;
        classification.num_examples += 1;
    }

    /// Takes a document and a label and tokenizes the document by
    /// breaking on whitespace characters. The document is added to the list
    /// of documents that the classifier is aware of and will train on next time
    /// the `train()` method is called 
    pub fn add_document(&mut self, document: &String, label: &String) {
        self.add_document_tokenized(&split_document(document), label);
    }

    /// Adds a list of (document, label) tuples to the classifier
    pub fn add_documents(&mut self, examples: &Vec<(String, String)>) {
        for &(ref document, ref label) in examples.iter() {
            self.add_document(document, label);
        }
    }

    /// Adds a list of (tokenized document, label) tuples to the classifier
    pub fn add_documents_tokenized(&mut self, examples: &Vec<(Vec<String>, String)>) {
        for &(ref document, ref label) in examples.iter() {
            self.add_document_tokenized(document, label);
        }
    }

    /// Gets a vector of all of the labels that the classifier has seen so far
    pub fn get_labels(&self) -> Vec<String> {
        let labels: Vec<String> =
            self.classifications.values().map(|c| c.label.clone()).collect();
        labels
    }

    /// Sets the [smoothing](http://en.wikipedia.org/wiki/Additive_smoothing)
    /// value (must be greater than 0.0)
    pub fn set_smoothing(&mut self, smoothing: f64) {
        if smoothing <= 0.0 {
            panic!("smoothing value must be a positive number");
        }
        self.smoothing = smoothing;
    }

    /// Trains the classifier on the documents that have been observed so far
    pub fn train(&mut self) {
        for (_, classification) in self.classifications.iter_mut() {
            classification.train(&self.vocab, self.num_examples, self.smoothing);
        }
    }

    /// Takes an unlabeled document that has been tokenized into a vector of strings
    /// and then computes a classifying label for the document
    pub fn classify_tokenized(&self, document: &Vec<String>) -> String {
        let mut max_score = f64::NEG_INFINITY;
        let mut max_classification = None;
        
        for classification in self.classifications.values() {
            let score = classification.score_document(document, &self.vocab);
            if score > max_score {
                max_classification = Some(classification);
                max_score = score;
            }
        }

        max_classification.expect("no classification found").label.clone()
    }

    /// Takes an unlabeled document and tokenizes it by breaking on spaces and
    /// then computes a classifying label for the document
    pub fn classify(&self, document: &String) -> String {
        self.classify_tokenized(&split_document(document))
    }

    /// Similar to classify but instead of returning a single label, returns all
    /// labels and the probabilities of each one given the document
    pub fn get_document_probabilities_tokenized(&self, document: &Vec<String>) -> Vec<(String, f64)> {        
        
        let all_probs:Vec<(String, f64)> = self.classifications.values().map(|classification| {
            let score = classification.score_document(document, &self.vocab);
            (classification.label.clone(), score)
        }).collect();

        let total_prob = all_probs.iter()
            .map(|&(_, s)| s)
            .fold(0.0, |acc, s| acc + s);

        all_probs.into_iter().map(|(c, s)| (c, 1.0 - s/total_prob) ).collect()
    }

    /// Similar to classify but instead of returning a single label, returns all
    /// labels and the probabilities of each one given the document
    pub fn get_document_probabilities(&self, document: &String) -> Vec<(String, f64)> {
        self.get_document_probabilities_tokenized(&split_document(document))
    }

    /// Encodes the classifier as a JSON string.
    pub fn to_json(&self) -> String {
        json::encode(self).ok().expect("encoding JSON failed")
    }

    /// Builds a new classifier from a JSON string
    pub fn from_json(encoded: &str) -> Classifier {
        let classifier: Classifier = json::decode(encoded).ok().expect("decoding JSON failed");
        classifier
    }

}


impl Classification {
    
    fn new(label: &String) -> Classification {
        Classification {
            label: label.clone(),
            num_examples: 0u32,
            num_words: 0u32,
            probability: 0.0f64,
            default_word_probability: 0.0f64,
            words: HashMap::new(),
        }
    }

    fn add_word(&mut self, word: &String) {
        self.num_words += 1;
        if self.words.contains_key(word) {
            self.words.get_mut(word).unwrap().0 += 1;
        } else {
            self.words.insert(word.clone(), (1, 0.0f64));
        }
    }

    // trains this classification
    fn train(&mut self, vocab: &HashSet<String>, total_examples: u32, smoothing: f64) {
        // the probability of this classification
        self.probability = self.num_examples as f64 / total_examples as f64;
        // the probability of any word that has not been seen in a document
        // labeled with this classification's label
        self.default_word_probability = smoothing /
            (self.num_words as f64 + smoothing * vocab.len() as f64);
        
        for word in vocab.iter() {
            if self.words.contains_key(word) {
                let word_entry = self.words.get_mut(word).unwrap();
                let word_count = word_entry.0;
                let p_word_given_label =
                    (word_count as f64 + smoothing) /
                    (self.num_words as f64 + smoothing * vocab.len() as f64);
                word_entry.1 = p_word_given_label;
            }
        }
    }

    // retrieves the probability of the document given this classification
    // times the probability of this classification
    fn score_document(&self, document: &Vec<String>, vocab: &HashSet<String>) -> f64 {
        let mut total = 0.0f64;
        for word in document.iter() {
            if vocab.contains(word) {
                let word_probability = match self.words.get(word) {
                    Some( &(_, p) ) => p,
                    None => self.default_word_probability,
                };
                total += word_probability.ln();
            }
        }
        self.probability.ln() + total
    }
}

// splits a String on whitespaces
fn split_document(document: &String) -> Vec<String> {
    let re = Regex::new(r"(\s)").unwrap();
    re.split(document).map(|s| s.to_string()).collect()
}