import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

from test import test_model


def train_model(model, dataset, device, config):
    # Set the number of training epochs
    num_epochs = config.num_epochs

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.l2_regularization)

    model.train()  # Set the model to training mode

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"log/{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        progress_bar = tqdm(dataset.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Write training loss and accuracy to TensorBoard
            iteration = epoch * len(dataset.train_loader) + batch_idx
            writer.add_scalar("Training Loss", train_loss / total_samples, iteration)
            writer.add_scalar("Training Accuracy", train_correct / total_samples, iteration)

            # Update the progress bar
            progress_bar.set_postfix({"Loss": train_loss / total_samples, "Accuracy": train_correct / total_samples})

        # Calculate epoch statistics
        train_loss /= total_samples
        train_acc = train_correct / total_samples

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        test_model(model, dataset.test_loader, device)

    writer.close()