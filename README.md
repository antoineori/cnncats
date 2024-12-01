# Inference API Service

This project provides an inference API for classifying images using a CNN model.

## File Structure
- **`models/`**:
   - `serialized_model.pt`: The serialized TorchScript model used for inference.
- **`test_pics/`**:
   - Contains example images for testing the API.
- **`cnn_model.py`**: CNN model definition.
- **`create_dummy_model.py`**: Script to generate a dummy model.
- **`inference_api.py`**: FastAPI-based inference service.
- **`Dockerfile`**: Dockerfile for containerizing the API.
- **`docker-compose.yml`**: Compose file for deploying the API.
- **`requirements.txt`**: Python dependencies for the project.
- **`serialize_model.py`**: Script to serialize a trained model.

## Requirements
- Python 3.8+
- Docker and Docker Compose

## Running Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
