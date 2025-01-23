import os
import ssl
import urllib.request

from ultralytics import YOLO

# Disable SSL verification (unsafe for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Define the model file path and URL to download the model if it doesn't exist
model_name = "yolo11s"
model_path = f"./yolo_models/{model_name}.pt"
model_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}.pt"

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Model file '{model_path}' not found. Downloading...")
    urllib.request.urlretrieve(model_url, model_path)
    print(f"Model downloaded to {model_path}")

# Load the model
model = YOLO(model_path)

# Train the model
# ref for train settings - https://docs.ultralytics.com/usage/cfg/#modes
train_results = model.train(
    data="yolo_train_config.yaml",  # path to dataset YAML
    epochs=200,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)


# TODO
# Data Augmentation Yolo Techniques - https://www.restack.io/p/data-augmentation-answer-yolo-cat-ai
# albumentations - https://albumentations.ai/docs/