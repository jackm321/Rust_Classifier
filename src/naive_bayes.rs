use std::collections::HashMap;
use std::f64;
use regex::Regex;
use rustc_serialize::json;

static DEFAULT_SMOOTHING: f64 = 1.0f64;

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
/// Naive Bayes classifier
pub struct Classifier {
    vocab: HashMap<String, u32>,
    num_examples: u32,
    smoothing: f64,
    trained: bool,
    classifications: HashMap<String, Classification>
}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
struct Classification {
    label: String,
    num_examples: u32,
    num_words: u32,
    probability: Option<f64>,
    default_word_probability: Option<f64>,
    words: HashMap<String, (u32, Option<f64>)>,
}

impl Classifier {
    
    /// Creates a new classifier
    pub fn new() -> Classifier {
        Classifier {
            vocab: HashMap::new(),
            num_examples: 0u32,
            smoothing: DEFAULT_SMOOTHING,
            trained: false,
            classifications: HashMap::new(),
        }
    }

    /// Takes a document that has been tokenized into a vector of strings
    /// and a label and adds the document to the list of documents that the
    /// classifier is aware of and will train on next time the `train()` method is called
    pub fn add_document_tokenized(&mut self, document: &Vec<String>, label: &String) {
        if document.len() == 0 { return; }

        self.num_examples += 1;
        self.trained = false;
        
        // make sure the classification already exists
        if !self.classifications.contains_key(label) {
            let c = Classification::new(label);
            self.classifications.insert(label.clone(), c);
        }

        let mut classification = self.classifications.get_mut(label).unwrap();
                
        for word in document.iter() {
            // add word to classification
            classification.add_word(word);
            
            // add word to vocab
            if self.vocab.contains_key(word) {
                *self.vocab.get_mut(word).unwrap() += 1;
            } else {
                self.vocab.insert(word.to_string(), 1);
            }
        }

        classification.num_examples += 1;
    }

    /// Takes a document and a label and tokenizes the document by
    /// breaking on whitespace characters. The document is added to the list
    /// of documents that the classifier is aware of and will train on next time the `train()` method is called 
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

    /// Sets the [smoothing](http://en.wikipedia.org/wiki/Additive_smoothing) value (must be greater than 0.0)
    pub fn set_smoothing(&mut self, smoothing: f64) {
        if smoothing <= 0.0 {
            panic!("smoothing must be a positive number");
        }
        self.smoothing = smoothing;
    }

    /// Trains the classifier on the documents that have been observed so far
    pub fn train(&mut self) {
        for (_, classification) in self.classifications.iter_mut() {
            classification.train(&self.vocab, self.num_examples, self.smoothing);
        }
        self.trained = true
    }

    /// Takes an unlabeled document that has been tokenized into a vector of strings
    //  and then computes a classifying label for the document
    pub fn classify_tokenized(&mut self, document: &Vec<String>) -> String {
        if !self.trained { self.train(); }

        let mut max_score = f64::NEG_INFINITY;
        let mut max_classification = None;
        
        for classification in self.classifications.values() {
            let score = classification.score_document(document, &self.vocab);
            if score > max_score {
                max_classification = Some(classification);
                max_score = score;
            }
        }

        max_classification.unwrap().label.clone()
    }

    /// Takes an unlabled document and tokenizes it by breaking on spaces and
    //  then computes a classifying label for the document
    pub fn classify(&mut self, document: &String) -> String {
        self.classify_tokenized(&split_document(document))
    }

    /// Similar to classify but instead of returning a single label, returns all
    /// labels and the probabilities of each one given the document
    pub fn get_document_probabilities_tokenized(&mut self, document: &Vec<String>) -> Vec<(String, f64)> {
        if !self.trained { self.train(); }
        
        self.classifications.values().map(|classification| {
            let score = classification.score_document(document, &self.vocab);
            (classification.label.clone(), score)
        }).collect()

    }

    /// Similar to classify but instead of returning a single label, returns all
    /// labels and the probabilities of each one given the document
    pub fn get_document_probabilities(&mut self, document: &String) -> Vec<(String, f64)> {
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
            probability: None,
            default_word_probability: None,
            words: HashMap::new(),
        }
    }

    fn add_word(&mut self, word: &String) {
        self.num_words += 1;
        if self.words.contains_key(word) {
            self.words.get_mut(word).unwrap().0 += 1;
        } else {
            self.words.insert(word.clone(), (1, None));
        }
    }

    fn train(&mut self, vocab: &HashMap<String, u32>, total_examples: u32, smoothing: f64) {
        self.probability = Some(self.num_examples as f64 / total_examples as f64);
        self.default_word_probability = Some(smoothing / (self.num_words as f64 + smoothing * vocab.len() as f64));
        
        for word in vocab.keys() {
            if self.words.contains_key(word) {
                let word_entry = self.words.get_mut(word).unwrap();
                let word_count = word_entry.0;
                let p_word_given_label = (word_count as f64 + smoothing) / (self.num_words as f64 + smoothing * vocab.len() as f64);
                word_entry.1 = Some(p_word_given_label);
            }
        }
    }

    fn score_document(&self, document: &Vec<String>, vocab: &HashMap<String, u32>) -> f64 {
        let mut total = 0.0f64;
        for word in document.iter() {
            if vocab.contains_key(word) {
                let word_probability = match self.words.get(word) {
                    Some( &(_, p) ) => p.unwrap(),
                    None => self.default_word_probability.unwrap(),
                };
                total += word_probability.ln();
            }
        }
        self.probability.unwrap().ln() + total
    }
}

fn split_document(document: &String) -> Vec<String> {
    let re = Regex::new(r"(\s)").unwrap();
    re.split(document).map(|s| s.to_string()).collect()
}