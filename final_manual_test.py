import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from log.Ualexnet.model import ImageClassifier
from log.Calexnet.model import ImageClassifier2
from log.Uresnet50.model import ImageClassifier3


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
image_dir = "../../data for emotion classification/dataset_cleaned/manual_testing/angry"

# List of model classes and their corresponding file paths
models = [
    {"class": ImageClassifier(num_classes=4), "path": "log/Ualexnet/alexnet.pth"},
    {"class": ImageClassifier3(num_classes=4), "path": "log/Uresnet50/resnet50.pth"},
    {"class": ImageClassifier2(num_classes=4), "path": "log/Calexnet/alexnet.pth"},
]

# Iterate over the models and test each one
for model_info in models:
    model_class = model_info["class"]
    model_path = model_info["path"]

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
                _, predicted_class = torch.max(output, 1)
                predicted_label = class_labels[predicted_class.item()]
                #output_name = output_names[predicted_class.item()]

            # Get the ground truth label from the folder name
            ground_truth_label = os.path.basename(root)

            # Check if the prediction matches the ground truth
            print(f"Model: {model_path}")
            print(f"Image: {image_file}")
            print("Predicted:  ", predicted_label)
            if predicted_label == ground_truth_label:
                correct_predictions += 1

            total_images += 1

            # Print the custom output names for the current model
            print(f"Folder: {ground_truth_label}")
            #print(f"Predicted animal (Current Model): {output_name}")
            print("----------------------------------")

    # Calculate the accuracy only if there are images
    if total_images != 0:
        accuracy = correct_predictions / total_images
        print(f"Accuracy: {accuracy * 100:.2f}%")
    else:
        print("No images found for evaluation.")

    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print("------------------------------")
