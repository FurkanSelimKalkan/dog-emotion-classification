import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ImageClassifier

# Load the first trained model
model_first = ImageClassifier(num_classes=4)
model_first.load_state_dict(torch.load("mobilenet26-05.pth", map_location=torch.device('cuda')))
model_first.eval()  # Set the model to evaluation mode

# Define the class labels and their corresponding custom output names
class_labels = ["angry", "happy", "relaxed", "sad"]

# Preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the directory containing the images
image_dir = "dataset_cleaned_angry/manual_testing"

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

        # Make predictions with the first model
        with torch.no_grad():
            output_first = model_first(input_image)
            _, predicted_class_first = torch.max(output_first, 1)
            predicted_label_first = class_labels[predicted_class_first.item()]
            #output_name_first = output_names[predicted_class_first.item()]

        # Get the ground truth label from the folder name
        ground_truth_label = os.path.basename(root)

        # Check if the prediction matches the ground truth
        print(f"Image: {image_file}")
        print("Predicted:  ", predicted_label_first)
        if predicted_label_first == ground_truth_label:
            correct_predictions += 1

        total_images += 1

        # Print the custom output names for both models
        print(f"Folder: {ground_truth_label}")
        #print(f"Predicted animal (First Model): {output_name_first}")
        print("----------------------------------")

# Calculate the accuracy only if there are images
if total_images != 0:
    accuracy = correct_predictions / total_images
    print(f"Accuracy: {accuracy * 100:.2f}%")
else:
    print("No images found for evaluation.")
