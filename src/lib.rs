//! A simple [naive Bayes classifier](http://en.wikipedia.org/wiki/Naive_Bayes_classifier)
//! crate for probabilistically classifying text documents.

//! ## Example: classifying documents about food
//!
//! ```rust
//! use classifier::NaiveBayes;
//! 
//! // create a new classifier
//! let mut nb = NaiveBayes::new();
//! 
//! // some example documents and labels
//! let examples = [
//! 
//!     ("beetroot water spinach okra water chestnut ricebean pea catsear courgette summer purslane. water spinach arugula pea tatsoi aubergine spring onion bush tomato kale radicchio turnip chicory salsify pea sprouts fava bean. dandelion zucchini burdock yarrow chickpea dandelion sorrel courgette turnip greens tigernut soybean radish artichoke wattle seed endive groundnut broccoli arugula.", "veggie"),
//! 
//!     ("sirloin meatloaf ham hock sausage meatball tongue prosciutto picanha turkey ball tip pastrami. ribeye chicken sausage, ham hock landjaeger pork belly pancetta ball tip tenderloin leberkas shank shankle rump. cupim short ribs ground round biltong tenderloin ribeye drumstick landjaeger short loin doner chicken shoulder spare ribs fatback boudin. pork chop shank shoulder, t-bone beef ribs drumstick landjaeger meatball.", "meat"),
//! 
//!     ("pea horseradish azuki bean lettuce avocado asparagus okra. kohlrabi radish okra azuki bean corn fava bean mustard tigernut jã­cama green bean celtuce collard greens avocado quandong fennel gumbo black-eyed pea. grape silver beet watercress potato tigernut corn groundnut. chickweed okra pea winter purslane coriander yarrow sweet pepper radish garlic brussels sprout groundnut summer purslane earthnut pea tomato spring onion azuki bean gourd. gumbo kakadu plum komatsuna black-eyed pea green bean zucchini gourd winter purslane silver beet rock melon radish asparagus spinach.", "veggie"),
//! 
//!     ("sirloin porchetta drumstick, pastrami bresaola landjaeger turducken kevin ham capicola corned beef. pork cow capicola, pancetta turkey tri-tip doner ball tip salami. fatback pastrami rump pancetta landjaeger. doner porchetta meatloaf short ribs cow chuck jerky pork chop landjaeger picanha tail.", "meat"),
//! 
//! ];
//! 
//! // add each example document to the classifier
//! for &(document, label) in examples.iter() {
//!     nb.add_document(&document.to_string(), &label.to_string());
//! }
//! 
//! // train the classifier
//! nb.train();
//! 
//! // now try to classify a new document with the classifier
//! let food_document = "salami pancetta beef ribs".to_string();
//! 
//! assert_eq!( nb.classify(&food_document), "meat" );
//! ```
//! meat examples from [baconipsum](http://baconipsum.com/), veggie examples from: [veggieipsum](http://veggieipsum.com/)

extern crate serde;
extern crate regex;

mod naive_bayes;
pub use self::naive_bayes::Classifier as NaiveBayes;