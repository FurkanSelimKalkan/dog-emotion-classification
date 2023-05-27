import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class ImageClassificationDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # Random rotation up to 10 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random changes in brightness and contrast
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translations
            transforms.RandomPerspective(distortion_scale=0.2),  # Random perspective changes
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = ImageFolder(self.data_dir + "/train/", transform=self.transform)
        self.val_dataset = ImageFolder(self.data_dir + "/val/", transform=self.transform)
        self.test_dataset = ImageFolder(self.data_dir + "/test/", transform=self.transform)
        self.num_classes = len(self.train_dataset.classes)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
