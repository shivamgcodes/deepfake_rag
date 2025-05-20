import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from torch.utils.data import DataLoader
from config import parentConfig as config
from setup_logging import logger
from tqdm import tqdm
from ImageDataset import CustomImageDataset

model_path = os.environ.get("MODEL_PATH", "vggface2")  # Default to "vggface2" if not set
threshold = float(os.environ.get("THRESHOLD", 0.5))  # Default to 0.5 if not set

DEVICE = config.device
model = InceptionResnetV1(
    pretrained="vggface2", # replace with the trained model path
    classify=True,
    num_classes=1,
    device=DEVICE
)

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

def detect_and_classify(image_path):
    img = Image.open(image_path).convert("RGB")

    # Detect face
    face = mtcnn(img)

    if face is None:
        print(f"No face detected in image: {image_path}")
        return None

    face = face.unsqueeze(0).to(DEVICE)  # Add batch dimension

    # Run classification
    with torch.no_grad():
        output = model(face)
        prob = torch.sigmoid(output).item()
        label = 1 if prob >= threshold else 0

    print(f"Image: {image_path} | Confidence: {prob:.4f} | Label: {label}")
    return label

