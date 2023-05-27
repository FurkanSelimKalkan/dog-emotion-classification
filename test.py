import torch


def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    test_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    test_acc = test_correct / total_samples
    print(f"Test Accuracy: {test_acc:.4f}")

