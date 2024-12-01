import torch
from cnn_model import CatCNN

# Configuration
MODEL_PATH = "models/best_model.pt"  # Path to the trained model weights
SCRIPTED_MODEL_PATH = "models/serialized_model.pt"  # Path to save the serialized model


def serialize_model():
    """Load the trained model and serialize it to TorchScript format."""
    # Initialize the model
    model = CatCNN()

    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    # Serialize the model to TorchScript format
    scripted_model = torch.jit.script(model)

    # Save the serialized model
    scripted_model.save(SCRIPTED_MODEL_PATH)
    print(f"Model serialized and saved to {SCRIPTED_MODEL_PATH}")

if __name__ == "__main__":
    serialize_model()
