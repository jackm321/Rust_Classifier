extern crate classifier;
use classifier::NaiveBayes;

#[test]
fn food_document_test() {
    
    // create a new classifier
    let mut nb = NaiveBayes::new();

    // some example documents and labels
    let examples = [

        ("beetroot water spinach okra water chestnut ricebean pea catsear courgette summer purslane. water spinach arugula pea tatsoi aubergine spring onion bush tomato kale radicchio turnip chicory salsify pea sprouts fava bean. dandelion zucchini burdock yarrow chickpea dandelion sorrel courgette turnip greens tigernut soybean radish artichoke wattle seed endive groundnut broccoli arugula.", "veggie"),

        ("sirloin meatloaf ham hock sausage meatball tongue prosciutto picanha turkey ball tip pastrami. ribeye chicken sausage, ham hock landjaeger pork belly pancetta ball tip tenderloin leberkas shank shankle rump. cupim short ribs ground round biltong tenderloin ribeye drumstick landjaeger short loin doner chicken shoulder spare ribs fatback boudin. pork chop shank shoulder, t-bone beef ribs drumstick landjaeger meatball.", "meat"),

        ("pea horseradish azuki bean lettuce avocado asparagus okra. kohlrabi radish okra azuki bean corn fava bean mustard tigernut jã­cama green bean celtuce collard greens avocado quandong fennel gumbo black-eyed pea. grape silver beet watercress potato tigernut corn groundnut. chickweed okra pea winter purslane coriander yarrow sweet pepper radish garlic brussels sprout groundnut summer purslane earthnut pea tomato spring onion azuki bean gourd. gumbo kakadu plum komatsuna black-eyed pea green bean zucchini gourd winter purslane silver beet rock melon radish asparagus spinach.", "veggie"),

        ("sirloin porchetta drumstick, pastrami bresaola landjaeger turducken kevin ham capicola corned beef. pork cow capicola, pancetta turkey tri-tip doner ball tip salami. fatback pastrami rump pancetta landjaeger. doner porchetta meatloaf short ribs cow chuck jerky pork chop landjaeger picanha tail.", "meat"),

    ];

    // add each example document to the classifier
    for &(document, label) in examples.iter() {
        nb.add_document(&document.to_string(), &label.to_string());
    }

    // train the classifier
    nb.train();

    // now try to classify a new sentence with the classifier
    let food_sentence = "salami pancetta beef ribs".to_string();

    assert_eq!( nb.classify(&food_sentence), "meat" );

    // export and reimport the classifier just to try it
    let nb2 = NaiveBayes::from_json( &nb.to_json() );
    assert_eq!( nb2.classify(&food_sentence), "meat" );

    // try getting all probabilities for the sentence
    let all_probs = nb.get_document_probabilities(&food_sentence);
    if all_probs[0].0 == "meat" {
        assert_eq!(all_probs[1].0, "veggie");
        assert!(all_probs[0].1 > all_probs[1].1);
    } else {
        assert_eq!(all_probs[1].0, "meat");
        assert!(all_probs[0].1 < all_probs[1].1);
    }

}

// test out methods that train from tokenized documents
#[test]
fn food_document_tokenized_test() {
    
    // create a new classifier
    let mut nb = NaiveBayes::new();

    // some example documents and labels
    let examples = [

        ("beetroot water spinach okra water chestnut ricebean pea catsear courgette summer purslane. water spinach arugula pea tatsoi aubergine spring onion bush tomato kale radicchio turnip chicory salsify pea sprouts fava bean. dandelion zucchini burdock yarrow chickpea dandelion sorrel courgette turnip greens tigernut soybean radish artichoke wattle seed endive groundnut broccoli arugula.".split(" ").map(|s| s.to_string()).collect(), "veggie"),

        ("sirloin meatloaf ham hock sausage meatball tongue prosciutto picanha turkey ball tip pastrami. ribeye chicken sausage, ham hock landjaeger pork belly pancetta ball tip tenderloin leberkas shank shankle rump. cupim short ribs ground round biltong tenderloin ribeye drumstick landjaeger short loin doner chicken shoulder spare ribs fatback boudin. pork chop shank shoulder, t-bone beef ribs drumstick landjaeger meatball.".split(" ").map(|s| s.to_string()).collect(), "meat"),

        ("pea horseradish azuki bean lettuce avocado asparagus okra. kohlrabi radish okra azuki bean corn fava bean mustard tigernut jã­cama green bean celtuce collard greens avocado quandong fennel gumbo black-eyed pea. grape silver beet watercress potato tigernut corn groundnut. chickweed okra pea winter purslane coriander yarrow sweet pepper radish garlic brussels sprout groundnut summer purslane earthnut pea tomato spring onion azuki bean gourd. gumbo kakadu plum komatsuna black-eyed pea green bean zucchini gourd winter purslane silver beet rock melon radish asparagus spinach.".split(" ").map(|s| s.to_string()).collect(), "veggie"),

        ("sirloin porchetta drumstick, pastrami bresaola landjaeger turducken kevin ham capicola corned beef. pork cow capicola, pancetta turkey tri-tip doner ball tip salami. fatback pastrami rump pancetta landjaeger. doner porchetta meatloaf short ribs cow chuck jerky pork chop landjaeger picanha tail.".split(" ").map(|s| s.to_string()).collect(), "meat"),

    ];

    // add each example document to the classifier
    for &(ref document, ref label) in examples.iter() {
        nb.add_document_tokenized(&document, &label.to_string());
    }

    // train the classifier
    nb.train();

    // now try to classify a new sentence with the classifier
    let food_sentence = "salami pancetta beef ribs".split(" ").map(|s| s.to_string()).collect();

    assert_eq!( nb.classify_tokenized(&food_sentence), "meat" );

    // try getting all probabilities for the sentence
    let all_probs = nb.get_document_probabilities_tokenized(&food_sentence);
    if all_probs[0].0 == "meat" {
        assert_eq!(all_probs[1].0, "veggie");
        assert!(all_probs[0].1 > all_probs[1].1);
    } else {
        assert_eq!(all_probs[1].0, "meat");
        assert!(all_probs[0].1 < all_probs[1].1);
    }

}

#[test]
fn food_documents_test() {
    
    // create a new classifier
    let mut nb = NaiveBayes::new();

    // some example documents and labels
    let examples = vec![

        ("beetroot water spinach okra water chestnut ricebean pea catsear courgette summer purslane. water spinach arugula pea tatsoi aubergine spring onion bush tomato kale radicchio turnip chicory salsify pea sprouts fava bean. dandelion zucchini burdock yarrow chickpea dandelion sorrel courgette turnip greens tigernut soybean radish artichoke wattle seed endive groundnut broccoli arugula.".to_string(), "veggie".to_string()),

        ("sirloin meatloaf ham hock sausage meatball tongue prosciutto picanha turkey ball tip pastrami. ribeye chicken sausage, ham hock landjaeger pork belly pancetta ball tip tenderloin leberkas shank shankle rump. cupim short ribs ground round biltong tenderloin ribeye drumstick landjaeger short loin doner chicken shoulder spare ribs fatback boudin. pork chop shank shoulder, t-bone beef ribs drumstick landjaeger meatball.".to_string(), "meat".to_string()),

        ("pea horseradish azuki bean lettuce avocado asparagus okra. kohlrabi radish okra azuki bean corn fava bean mustard tigernut jã­cama green bean celtuce collard greens avocado quandong fennel gumbo black-eyed pea. grape silver beet watercress potato tigernut corn groundnut. chickweed okra pea winter purslane coriander yarrow sweet pepper radish garlic brussels sprout groundnut summer purslane earthnut pea tomato spring onion azuki bean gourd. gumbo kakadu plum komatsuna black-eyed pea green bean zucchini gourd winter purslane silver beet rock melon radish asparagus spinach.".to_string(), "veggie".to_string()),

        ("sirloin porchetta drumstick, pastrami bresaola landjaeger turducken kevin ham capicola corned beef. pork cow capicola, pancetta turkey tri-tip doner ball tip salami. fatback pastrami rump pancetta landjaeger. doner porchetta meatloaf short ribs cow chuck jerky pork chop landjaeger picanha tail.".to_string(), "meat".to_string()),

    ];

    nb.add_documents(&examples);

    nb.train();

    // now try to classify a new sentence with the classifier
    let food_sentence = "salami pancetta beef ribs".to_string();

    assert_eq!( nb.classify(&food_sentence), "meat" );

    // try getting all probabilities for the sentence
    let all_probs = nb.get_document_probabilities(&food_sentence);
    if all_probs[0].0 == "meat" {
        assert_eq!(all_probs[1].0, "veggie");
        assert!(all_probs[0].1 > all_probs[1].1);
    } else {
        assert_eq!(all_probs[1].0, "meat");
        assert!(all_probs[0].1 < all_probs[1].1);
    }

}


#[test]
fn get_labels_test() {

    // create a new classifier
    let mut nb = NaiveBayes::new();

    // some example documents and labels
    let examples = [

        ("beetroot water spinach okra water chestnut ricebean pea catsear courgette summer purslane. water spinach arugula pea tatsoi aubergine spring onion bush tomato kale radicchio turnip chicory salsify pea sprouts fava bean. dandelion zucchini burdock yarrow chickpea dandelion sorrel courgette turnip greens tigernut soybean radish artichoke wattle seed endive groundnut broccoli arugula.", "veggie"),

        ("sirloin meatloaf ham hock sausage meatball tongue prosciutto picanha turkey ball tip pastrami. ribeye chicken sausage, ham hock landjaeger pork belly pancetta ball tip tenderloin leberkas shank shankle rump. cupim short ribs ground round biltong tenderloin ribeye drumstick landjaeger short loin doner chicken shoulder spare ribs fatback boudin. pork chop shank shoulder, t-bone beef ribs drumstick landjaeger meatball.", "meat"),

        ("pea horseradish azuki bean lettuce avocado asparagus okra. kohlrabi radish okra azuki bean corn fava bean mustard tigernut jã­cama green bean celtuce collard greens avocado quandong fennel gumbo black-eyed pea. grape silver beet watercress potato tigernut corn groundnut. chickweed okra pea winter purslane coriander yarrow sweet pepper radish garlic brussels sprout groundnut summer purslane earthnut pea tomato spring onion azuki bean gourd. gumbo kakadu plum komatsuna black-eyed pea green bean zucchini gourd winter purslane silver beet rock melon radish asparagus spinach.", "veggie"),

        ("sirloin porchetta drumstick, pastrami bresaola landjaeger turducken kevin ham capicola corned beef. pork cow capicola, pancetta turkey tri-tip doner ball tip salami. fatback pastrami rump pancetta landjaeger. doner porchetta meatloaf short ribs cow chuck jerky pork chop landjaeger picanha tail.", "meat"),

    ];

    // test before adding documents
    assert!(nb.get_labels().len() == 0);

    // add each example document to the classifier
    for &(document, label) in examples.iter() {
        nb.add_document(&document.to_string(), &label.to_string());
    }

    // test after adding documents
    let labels = nb.get_labels();
    assert!(
        (labels[0] == "meat" && labels[1] == "veggie") ||
        (labels[0] == "veggie" && labels[1] == "meat") );

}

#[test]
fn food_smoothing_test() {
    
    // create a new classifier
    let mut nb = NaiveBayes::new();

    // some example documents and labels
    let examples = [

        ("beetroot water spinach okra water chestnut ricebean pea catsear courgette summer purslane. water spinach arugula pea tatsoi aubergine spring onion bush tomato kale radicchio turnip chicory salsify pea sprouts fava bean. dandelion zucchini burdock yarrow chickpea dandelion sorrel courgette turnip greens tigernut soybean radish artichoke wattle seed endive groundnut broccoli arugula.", "veggie"),

        ("sirloin meatloaf ham hock sausage meatball tongue prosciutto picanha turkey ball tip pastrami. ribeye chicken sausage, ham hock landjaeger pork belly pancetta ball tip tenderloin leberkas shank shankle rump. cupim short ribs ground round biltong tenderloin ribeye drumstick landjaeger short loin doner chicken shoulder spare ribs fatback boudin. pork chop shank shoulder, t-bone beef ribs drumstick landjaeger meatball.", "meat"),

        ("pea horseradish azuki bean lettuce avocado asparagus okra. kohlrabi radish okra azuki bean corn fava bean mustard tigernut jã­cama green bean celtuce collard greens avocado quandong fennel gumbo black-eyed pea. grape silver beet watercress potato tigernut corn groundnut. chickweed okra pea winter purslane coriander yarrow sweet pepper radish garlic brussels sprout groundnut summer purslane earthnut pea tomato spring onion azuki bean gourd. gumbo kakadu plum komatsuna black-eyed pea green bean zucchini gourd winter purslane silver beet rock melon radish asparagus spinach.", "veggie"),

        ("sirloin porchetta drumstick, pastrami bresaola landjaeger turducken kevin ham capicola corned beef. pork cow capicola, pancetta turkey tri-tip doner ball tip salami. fatback pastrami rump pancetta landjaeger. doner porchetta meatloaf short ribs cow chuck jerky pork chop landjaeger picanha tail.", "meat"),

    ];

    // try setting smoothing
    nb.set_smoothing(0.1);

    // add each example document to the classifier
    for &(document, label) in examples.iter() {
        nb.add_document(&document.to_string(), &label.to_string());
    }

    // train the classifier
    nb.train();

    // now try to classify a new sentence with the classifier
    let food_sentence = "salami pancetta beef ribs".to_string();

    assert_eq!( nb.classify(&food_sentence), "meat" );

    // try getting all probabilities for the sentence
    let all_probs = nb.get_document_probabilities(&food_sentence);
    if all_probs[0].0 == "meat" {
        assert_eq!(all_probs[1].0, "veggie");
        assert!(all_probs[0].1 > all_probs[1].1);
    } else {
        assert_eq!(all_probs[1].0, "meat");
        assert!(all_probs[0].1 < all_probs[1].1);
    }

}

