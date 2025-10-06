from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_PATH = "task2_ner_image_pipeline/ner_model/ner_model_out"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

# Create pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Predefined dataset classes
ANIMAL_CLASSES = ["dog", "horse", "elephant", "butterfly", "chicken", 
                  "cat", "cow", "sheep", "spider", "squirrel"]

def extract_animals(text: str):
    """
    Extract animal names from text using NER + fallback keyword search.
    """
    results = ner_pipeline(text)

    # Collect model outputs
    entities = []
    for r in results:
        if "word" in r:
            entities.append(r["word"].lower())
        if "entity_group" in r:
            entities.append(r["entity_group"].lower())

    # Match against known classes
    detected = [a for a in ANIMAL_CLASSES if any(a in e for e in entities)]

    # Fallback: search directly in the text (singular/plural)
    if not detected:
        text_lower = text.lower()
        for a in ANIMAL_CLASSES:
            if a in text_lower or a + "s" in text_lower:  # handle plural
                detected.append(a)

    return list(set(detected))