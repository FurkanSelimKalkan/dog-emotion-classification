from dataset import ImageClassificationDataset
from model import ImageClassifier
import train
import test
import torch
from torchvision import transforms
from config import Config

if __name__ == "__main__":
    # Load the configuration settings
    config = Config()

    # Initialize the dataset with data augmentation
    data_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize the dataset
    dataset = ImageClassificationDataset(config.data_dir)

    # Set the device (CPU or GPU)
    device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = ImageClassifier(num_classes=config.num_classes).to(device)

    # Set the number of CPU workers for data loading
    num_workers = config.num_cpu_workers if device.type == "cpu" else 0

    # Set the batch size for data loaders
    dataset.train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    dataset.val_loader = torch.utils.data.DataLoader(
        dataset.val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    dataset.test_loader = torch.utils.data.DataLoader(
        dataset.test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Train the model
    train.train_model(model, dataset, device, config)

    # Save the trained model
    torch.save(model.state_dict(), "cleanedDataAlexPre8-05.pth")

    # Test the model
    test.test_model(model, dataset.test_loader, device)
