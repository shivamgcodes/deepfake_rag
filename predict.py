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
