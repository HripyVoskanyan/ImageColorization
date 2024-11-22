from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_split, transform=None):
        self.dataset_split = dataset_split
        self.transform = transform

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, idx):
        image = self.dataset_split[idx]["image"]
        image = image.convert("RGB")  # Ensure all images are RGB
        if self.transform:
            transformed_image = self.transform(image)
            grayscale_image = transformed_image.mean(dim=0, keepdim=True)  # Convert to grayscale
            return grayscale_image, transformed_image  # Input and target
        return image

def load_data(batch_size=32, val_split=0.2, test_split=0.1):
    # Load the Tiny ImageNet dataset
    dataset = load_dataset("zh-plus/tiny-imagenet")["train"]

    # Calculate sizes for splits
    test_size = int(test_split * len(dataset))
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size - test_size

    # Split the dataset
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(128, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    # Simple transform for validation/test
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Prepare datasets
    train_dataset = TinyImageNetDataset(dataset_split=train_data, transform=train_transform)
    val_dataset = TinyImageNetDataset(dataset_split=val_data, transform=eval_transform)
    test_dataset = TinyImageNetDataset(dataset_split=test_data, transform=eval_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
