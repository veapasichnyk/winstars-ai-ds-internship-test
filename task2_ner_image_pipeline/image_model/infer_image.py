import torch
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

# Load model & class names
checkpoint = torch.load("resnet_animals.pth", map_location=DEVICE)
class_names = checkpoint["class_names"]

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

#  Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_image(image_path):
    """
    Predict animal class from an input image.
    Args:
        image_path (str): path to image
    Returns:
        str: predicted class name (English)
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

if __name__ == "__main__":
    test_image = "animals10/raw-img/gatto/1.jpg"
    print("Prediction:", predict_image(test_image))