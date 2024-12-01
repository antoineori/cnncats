import torch
from cnn_model import CatCNN

# Path to save the dummy model
DUMMY_MODEL_PATH = "models/serialized_model.pt"


def create_dummy_model():
    """Create a dummy model with random weights and serialize it."""
    model = CatCNN()  # Initialize the model
    model.eval()  # Set to evaluation mode

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
    output = model(dummy_input)  # Forward pass
    print(f"Output shape: {output.shape}")  # Print the shape of the output

    # Serialize the model to TorchScript format
    scripted_model = torch.jit.script(model)
    scripted_model.save(DUMMY_MODEL_PATH)
    print(f"Dummy model saved to {DUMMY_MODEL_PATH}")


if __name__ == "__main__":
    create_dummy_model()
