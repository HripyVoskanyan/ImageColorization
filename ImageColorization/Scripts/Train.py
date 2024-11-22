import os
from CNN import ColorizationCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

# Define your custom dataset
class GrayscaleDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        grayscale_img = Image.open(img_path).convert("L")
        if self.transform:
            grayscale_img = self.transform(grayscale_img)
        rgb_img = grayscale_img.expand(3, -1, -1)  # Expand grayscale tensor to 3 channels
        return grayscale_img, rgb_img

