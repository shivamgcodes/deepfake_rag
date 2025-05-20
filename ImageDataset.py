import cv2
from PIL import Image
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        real = os.path.join(img_dir, "Real")
        fake = os.path.join(img_dir, "Fake")
        # loading the entire dataset into the memory
        # this is not recommended for large datasets
        # but for this dataset it is ok
        for image_file in tqdm(os.listdir(real)):
            path = os.path.join(real, image_file)
            img = Image.open(path).convert("RGB")
            self.images.append(img)
            self.labels.append(1)

        for image_file in tqdm(os.listdir(fake)):
            path = os.path.join(fake, image_file)
            img = Image.open(path).convert("RGB")
            self.images.append(img)
            self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

