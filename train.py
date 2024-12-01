import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_model import CatCNN
import wandb

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 20
DATA_DIR = "dataset"  # Path to dataset
MODEL_SAVE_PATH = "models/best_model.pt"

# Initialize Weights & Biases
wandb.init(project="cat-classifier", config={
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS
})

def load_data(data_dir):
    """Load and preprocess the dataset."""
    # Data augmentation for training
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    # Preprocessing for validation and testing
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=test_transforms)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device):
    """Train the model."""
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log training loss
        wandb.log({"train/loss": running_loss / len(train_loader)})

        # Validate the model
        val_accuracy = validate_model(model, val_loader, device)

        # Log validation accuracy
        wandb.log({"validate/accuracy": val_accuracy})

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Best model saved!")

def validate_model(model, val_loader, device):
    """Validate the model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = load_data(DATA_DIR)

    # Initialize the model
    model = CatCNN().to(device)

    # Train the model
    train_model(model, train_loader, val_loader, device)
