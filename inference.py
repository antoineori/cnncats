import torch
from torchvision import transforms
from PIL import Image

# Configuration
SCRIPTED_MODEL_PATH = "models/serialized_model.pt"  # Path to the serialized model
CLASS_NAMES = ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7"]  # Replace with actual class names

def load_model():
    """Load the serialized model."""
    model = torch.jit.load(SCRIPTED_MODEL_PATH)
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess the image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(image_path):
    """Make a prediction on the given image."""
    model = load_model()
    image = preprocess_image(image_path)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Return the class name
    return CLASS_NAMES[predicted.item()]

if __name__ == "__main__":
    # Test inference with a sample image
    SAMPLE_IMAGE_PATH = "sample.jpg"  # Replace with the path to a sample image
    result = predict(SAMPLE_IMAGE_PATH)
    print(f"Predicted Class: {result}")
