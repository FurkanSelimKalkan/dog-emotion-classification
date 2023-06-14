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
image_dir = "../../data for emotion classification/dataset_cleaned/manual_testing/relaxed"
#image_dir = "../../data for emotion classification/dataset_cleaned/test/relaxed"

# List of model classes, their corresponding file paths, and softmax functions
models = [
    {"class": ImageClassifier(num_classes=4), "path": "log/Ualexnet/alexnet.pth", "name": "Uncleaned Alexnet"},
    {"class": ImageClassifier3(num_classes=4), "path": "log/Uresnet50/resnet50.pth", "name": "Uncleaned ResNet50"},
    {"class": ImageClassifier2(num_classes=4), "path": "log/Calexnet/alexnet.pth", "name": "Cleaned Alexnet30"},
    {"class": ImageClassifier4(num_classes=4), "path": "log/Calexnet70Epochs/alexnet70.pth", "name": "Cleaned Alexnet70"},
    {"class": ImageClassifier5(num_classes=4), "path": "log/Calexnet100Epochs/alexnet100.pth", "name": "Cleaned Alexnet100"},
]

correct_predictions = 0
total_images = 0

if len(models) == 0:
    print("No models found. Please check the 'models' list.")
    exit()

# Iterate over the images and perform predictions
print(f"Using device: {device}")
for root, dirs, files in os.walk(image_dir):
    for image_file in files:
        image_path = os.path.join(root, image_file)

        # Load and preprocess the input image
        image = Image.open(image_path)
        input_image = preprocess(image)
        input_image = input_image.unsqueeze(0)  # Add a batch dimension
        input_image = input_image.to(device)

        # Calculate average confidence values for each model
        average_confidence_values = [0] * len(models)

        for model_idx, model_info in enumerate(models):
            model_class = model_info["class"]
            model_path = model_info["path"]
            model_name = model_info["name"]

            # Load the model
            model = model_class
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()  # Set the model to evaluation mode

            # Make predictions with the current model
            with torch.no_grad():
                output = model(input_image)
                softmax = torch.nn.Softmax(dim=1)
                probabilities = softmax(output)

            # Calculate the confidence value for each predicted class
            confidence_values = probabilities[0].tolist()

            # Update average confidence values
            for i, confidence in enumerate(confidence_values):
                average_confidence_values[i] += confidence

            # Get the predicted label
            predicted_index = torch.argmax(probabilities, dim=1).item()
            predicted_label = class_labels[predicted_index]

            # Print the predicted label and confidence values for the current model
            print(f"Model: {model_name}")
            print(f"Predicted: {predicted_label}")
            print(f"Image: {image_file}")
            for i, emotion in enumerate(class_labels):
                print(f"{emotion}: {confidence_values[i] * 100:.2f}%")
            print("---------------------------")

        # Determine the final prediction based on the highest average confidence value
        max_average_confidence = max(average_confidence_values)
        final_prediction = class_labels[average_confidence_values.index(max_average_confidence)]

        ground_truth_label = os.path.basename(root)
        if (final_prediction == ground_truth_label):
            correct_predictions +=1

        total_images +=1

        # Print the final prediction for the image
        print(f"Image: {image_file}")
        print(f"Final Prediction: {final_prediction}")
        print("---------------------------")

    accuracy = correct_predictions / total_images
    print(f"Accuracy: {accuracy * 100:.2f}%")

print("Prediction process completed.")
