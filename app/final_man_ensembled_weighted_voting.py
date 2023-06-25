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
#image_dir = "../../data for emotion classification/dataset_cleaned/manual_testing/sad"
image_dir = "../../data for emotion classification/dataset_cleaned/test/happy"

# List of model classes, their corresponding file paths, and softmax functions
models = [
    {"class": ImageClassifier(num_classes=4), "path": "log/Ualexnet/alexnet.pth", "name": "Uncleaned Alexnet"},
    {"class": ImageClassifier3(num_classes=4), "path": "log/Uresnet50/resnet50.pth", "name": "Uncleaned ResNet50"},
    {"class": ImageClassifier2(num_classes=4), "path": "log/Calexnet/alexnet.pth", "name": "Cleaned Alexnet30"},
    {"class": ImageClassifier4(num_classes=4), "path": "log/Calexnet70Epochs/alexnet70.pth", "name": "Cleaned Alexnet70"},
    {"class": ImageClassifier5(num_classes=4), "path": "log/Calexnet100Epochs/alexnet100.pth", "name": "Cleaned Alexnet100"},
]

# Initialize model performance tracking dictionaries
model_performance = {
    model["name"]: {"correct_predictions": 0, "total_predictions": 0,
                    "correct_confidence_sum": 0.0, "incorrect_confidence_sum": 0.0}
    for model in models
}

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

        # Dictionary to hold the sum of confidence values for each class
        confidence_sums = {label: 0.0 for label in class_labels}

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

            # Get the predicted label and confidence
            predicted_index = torch.argmax(probabilities, dim=1).item()
            predicted_label = class_labels[predicted_index]
            predicted_confidence = probabilities[0][predicted_index].item()

            # Update confidence_sums
            confidence_sums[predicted_label] += predicted_confidence

            # Determine the final prediction based on the class with the highest sum of confidence values
            final_prediction = max(confidence_sums, key=confidence_sums.get)

            # Update model performance tracking
            ground_truth_label = os.path.basename(root)
            if predicted_label == ground_truth_label:
                model_performance[model_name]["correct_predictions"] += 1
                model_performance[model_name]["correct_confidence_sum"] += predicted_confidence
            else:
                model_performance[model_name]["incorrect_confidence_sum"] += predicted_confidence
            model_performance[model_name]["total_predictions"] += 1

            # Print the predicted label and confidence values for the current model
            print(f"Model: {model_name}")
            print(f"Predicted: {predicted_label}")
            print(f"Image: {image_file}")
            for i, emotion in enumerate(class_labels):
                print(f"{emotion}: {probabilities[0][i].item() * 100:.2f}%")
            print("---------------------------")

        # Determine the final prediction based on the class with the highest sum of confidence values
        final_prediction = max(confidence_sums, key=confidence_sums.get)

        if (final_prediction == ground_truth_label):
            correct_predictions +=1

        total_images +=1

        # Print the final prediction for the image
        print(f"Image: {image_file}")
        print(f"Final Prediction: {final_prediction}")
        print("---------------------------")

    accuracy = correct_predictions / total_images
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Print performance for each model
for model_name, performance in model_performance.items():
    correct_percentage = performance["correct_predictions"] / performance["total_predictions"] * 100
    avg_correct_confidence = performance["correct_confidence_sum"] / performance["correct_predictions"]
    avg_incorrect_confidence = performance["incorrect_confidence_sum"] / (performance["total_predictions"] - performance["correct_predictions"])
    print(f"Model: {model_name}")
    print(f"Correct prediction rate: {correct_percentage:.2f}%")
    print(f"Average confidence when correct: {avg_correct_confidence:.2f}")
    print(f"Average confidence when incorrect: {avg_incorrect_confidence:.2f}")
    print("---------------------------")

print("Prediction process completed.")
