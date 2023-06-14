import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from log.Ualexnet.model import ImageClassifier
from log.Calexnet.model import ImageClassifier2
from log.Uresnet50.model import ImageClassifier3
from log.Calexnet70Epochs.model import ImageClassifier4
from log.Calexnet100Epochs.model import ImageClassifier5

# Check for CUDA availability and define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the class labels and their corresponding custom output names
class_labels = ["angry", "happy", "relaxed", "sad"]

# Preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the directory containing the images
# image_dir = "../../data for emotion classification/dataset_cleaned/test"
image_dir = "../../data for emotion classification/dataset_cleaned/manual_testing/happy"

# List of model classes, their corresponding file paths, and softmax functions
models = [
    {"class": ImageClassifier(num_classes=4), "path": "log/Ualexnet/alexnet.pth", "name": "Uncleaned Alexnet"},
    {"class": ImageClassifier3(num_classes=4), "path": "log/Uresnet50/resnet50.pth", "name": "Uncleaned ResNet50"},
    {"class": ImageClassifier2(num_classes=4), "path": "log/Calexnet/alexnet.pth", "name": "Cleaned Alexnet30"},
    {"class": ImageClassifier4(num_classes=4), "path": "log/Calexnet70Epochs/alexnet70.pth", "name": "Cleaned Alexnet70"},
    {"class": ImageClassifier5(num_classes=4), "path": "log/Calexnet100Epochs/alexnet100.pth", "name": "Cleaned Alexnet100"},
]

# Iterate over the models and test each one
for model_info in models:
    model_class = model_info["class"]
    model_path = model_info["path"]
    model_name = model_info["name"]

    # Load the model
    model = model_class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Variables to keep track of correct predictions
    total_images = 0
    correct_predictions = 0

    # Recursively iterate over the images in the directory and its subdirectories
    for root, dirs, files in os.walk(image_dir):
        for image_file in files:
            image_path = os.path.join(root, image_file)
            # Load and preprocess the input image
            image = Image.open(image_path)
            input_image = preprocess(image)
            input_image = input_image.unsqueeze(0)  # Add a batch dimension
            input_image = input_image.to(device)

            # Make predictions with the current model
            with torch.no_grad():
                output = model(input_image)
                softmax = torch.nn.Softmax(dim=1)
                probabilities = softmax(output)

            # Get the ground truth label from the folder name
            ground_truth_label = os.path.basename(root)

            # Calculate the confidence value for each predicted class
            confidence_values = probabilities[0].tolist()

            # Check if the prediction matches the ground truth
            if class_labels[torch.argmax(probabilities, dim=1).item()] == ground_truth_label:
                correct_predictions += 1

            total_images += 1

            # Calculate the confidence value for each predicted class
            confidence_values = probabilities[0].tolist()

            # Get the predicted label
            predicted_index = torch.argmax(probabilities, dim=1).item()
            predicted_label = class_labels[predicted_index]

            # Print the predicted label
            print(f"Model:  {model_name}")
            print(f"Predicted: {predicted_label}")
            print(f"To Predict: {ground_truth_label}")

            # Print the confidence values for each emotion
            print(f"Image: {image_file}")
            for i, emotion in enumerate(class_labels):
                print(f"{emotion}: {confidence_values[i] * 100:.2f}%")

            print("---------------------------")

    # Calculate the accuracy
    if total_images != 0:
        accuracy = correct_predictions / total_images
        print(f"Model: {model_path}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"Model: {model_path}")
        print("No images found for evaluation.")

    print(f"Device: {device}")
    print("------------------------------")
