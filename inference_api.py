import torch
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image

# Configuration
SCRIPTED_MODEL_PATH = "models/serialized_model.pt"  # Path to the serialized model
CLASS_NAMES = ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7"]  # Adjust for binary classification

# Load the serialized model
model = torch.jit.load(SCRIPTED_MODEL_PATH)
model.eval()

# Initialize FastAPI app
app = FastAPI()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

def preprocess_image(image_file):
    """Preprocess the uploaded image."""
    image = Image.open(image_file).convert("RGB")
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file and return a prediction.
    :param file: Uploaded image file.
    :return: Prediction result (e.g., 'cat' or 'not cat').
    """
    try:
        # Read and preprocess the image
        image_tensor = preprocess_image(file.file)

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[predicted.item()]

        return {"class": predicted_class}

    except Exception as e:
        return {"error": str(e)}

