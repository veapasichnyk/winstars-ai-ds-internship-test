import torch
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
from PIL import Image
import matplotlib.pyplot as plt

# CONFIG
NER_MODEL_PATH = "task2_ner_image_pipeline/ner_model/ner_model_out"
IMAGE_MODEL_PATH = "resnet_animals.pth"   # saved from train_image.py
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset classes (English)
ANIMAL_CLASSES = [
    "dog", "horse", "elephant", "butterfly", "chicken",
    "cat", "cow", "sheep", "spider", "squirrel"
]

# LOAD NER
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
ner_pipeline = hf_pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_animals(text: str):
    """Extract animal entities from text using fine-tuned NER model."""
    results = ner_pipeline(text)
    entities = [ent["word"].lower() for ent in results]
    detected = [a for a in ANIMAL_CLASSES if a in entities]

    if not detected:  # fallback search
        text_lower = text.lower()
        detected = [a for a in ANIMAL_CLASSES if a in text_lower]

    return detected

# LOAD IMAGE CLASSIFIER
checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def classify_image(image_path: str):
    """Predict animal class from image using trained ResNet."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

def text_image_pipeline(text: str, image_path: str) -> bool:
    """Pipeline: text + image → True/False."""
    text_animals = extract_animals(text)
    if not text_animals:
        print("No animals detected in text!")
        return False

    image_prediction = classify_image(image_path)

    print(f" Text entities: {text_animals}")
    print(f" Image prediction: {image_prediction}")

    return image_prediction.lower() in [a.lower() for a in text_animals]

def demo_case(text, image_path):
    """Show image, text, and pipeline result together."""
    result = text_image_pipeline(text, image_path)

    # Display text and result
    print(f" Text: {text}")
    print(f" Match result: {result}")

    # Display image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Text: {text} | Match: {result}")
    plt.show()

# Local test (optional)
if __name__ == "__main__":
    demo_case("There is a cat in the picture.", "/content/animals10/raw-img/gatto/1.jpg")