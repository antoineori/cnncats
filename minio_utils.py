from minio import Minio
from minio.error import S3Error

# Configuration
MINIO_URL = "localhost:9000"  # Replace with your MinIO server address
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "model-bucket"
MODEL_PATH = "serialized_model.pt"

# Initialize MinIO client
client = Minio(
    MINIO_URL,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False  # Set to True if using HTTPS
)

def upload_model():
    """Upload the serialized model to MinIO."""
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

    client.fput_object(
        BUCKET_NAME, "serialized_model.pt", MODEL_PATH
    )
    print(f"Model uploaded to bucket '{BUCKET_NAME}' as 'serialized_model.pt'.")

def download_model(download_path):
    """Download the serialized model from MinIO."""
    client.fget_object(
        BUCKET_NAME, "serialized_model.pt", download_path
    )
    print(f"Model downloaded to {download_path}.")

if __name__ == "__main__":
    # Test uploading and downloading
    upload_model()
    download_model("downloaded_model.pt")
