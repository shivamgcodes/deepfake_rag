import torch
from torchvision import datasets, transforms, models
import os
class parentConfig:
    validation_dir = "data/deepfake-and-real-images/Dataset/Validation/"
    validation_subset_percentage = None
    train_dir = "data/deepfake-and-real-images/Dataset/Train/"
    train_subset_percentage = None
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    num_workers = 4
    device ='cuda:0' if torch.cuda.is_available() else 'cpu'
    plots_dir = "/kaggle/working/plots/"
    logs_dir = "/kaggle/working/logs/"
    model_dir = "/kaggle/working/model/"
    transform =transforms.Compose([
    transforms.Resize(299),  # InceptionResNetV1 input size
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    def __init__(self):
        # Create directories if they don't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
