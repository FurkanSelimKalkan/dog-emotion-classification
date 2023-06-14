import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import torch
import torchvision.transforms as transforms
from log.Ualexnet.model import ImageClassifier
from log.Calexnet.model import ImageClassifier2
from log.Uresnet50.model import ImageClassifier3
from log.Calexnet70Epochs.model import ImageClassifier4
from log.Calexnet100Epochs.model import ImageClassifier5

load_dotenv()
host = os.getenv("HOST")

# Load the models
models = [
    {"model": ImageClassifier(num_classes=4), "path": "log/Ualexnet/alexnet.pth", "name": "Uncleaned Alexnet"},
    {"model": ImageClassifier3(num_classes=4), "path": "log/Uresnet50/resnet50.pth", "name": "Uncleaned ResNet50"},
    {"model": ImageClassifier2(num_classes=4), "path": "log/Calexnet/alexnet.pth", "name": "Cleaned Alexnet30"},
    {"model": ImageClassifier4(num_classes=4), "path": "log/Calexnet70Epochs/alexnet70.pth", "name": "Cleaned Alexnet70"},
    {"model": ImageClassifier5(num_classes=4), "path": "log/Calexnet100Epochs/alexnet100.pth", "name": "Cleaned Alexnet100"},
]

# Load model weights
for model_info in models:
    model_info["model"].load_state_dict(torch.load(model_info["path"], map_location=torch.device('cuda')))
    model_info["model"].eval()

# Define the class labels and their corresponding custom output names
class_labels = ["angry", "happy", "relaxed", "sad"]

# Preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def predict_image(input_image):
    # Dictionary to hold the sum of confidence values for each class
    confidence_sums = {label: 0.0 for label in class_labels}

    for model_info in models:
        with torch.no_grad():
            output = model_info["model"](input_image)
            softmax = torch.nn.Softmax(dim=1)
            probabilities = softmax(output)

        # Get the predicted label and confidence
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_confidence = probabilities[0][predicted_index].item()

        # Print the predicted label and confidence values for the current model
      #  print(f"Model: {model_info['path']}")
      #  print(f"Predicted: {class_labels[predicted_index]}")
      #  for i, emotion in enumerate(class_labels):
        #    print(f"{emotion}: {probabilities[0][i].item() * 100:.2f}%")
       # print("---------------------------")

        # Update confidence_sums
        confidence_sums[class_labels[predicted_index]] += predicted_confidence

    # Determine the final prediction based on the class with the highest sum of confidence values
    final_prediction = max(confidence_sums, key=confidence_sums.get)
    average_confidence = confidence_sums[final_prediction] / len(models)

    return final_prediction, average_confidence



@app.route('/predict', methods=['POST'])
def predict():
    print("Incoming request")
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image = Image.open(request.files['image'])
    input_image = preprocess(image)
    input_image = input_image.unsqueeze(0)  # Add a batch dimension

    final_prediction, average_confidence = predict_image(input_image)

    return jsonify({'prediction': final_prediction, 'confidence': average_confidence}), 200

@app.route('/test', methods=['get'])
def test():
    print("Incoming request")

    return jsonify({'API WORKS'}), 200

@app.route('/predict-multiple', methods=['POST'])
def predict_multiple():
    if 'images[]' not in request.files:
        return jsonify({'error': 'No image files uploaded'}), 400

    images = request.files.getlist('images[]')
    predictions = []

    for image in images:
        image = Image.open(image)
        input_image = preprocess(image)
        input_image = input_image.unsqueeze(0)  # Add a batch dimension

        prediction = predict_image(input_image)

        predictions.append(prediction)

    return jsonify({'predictions': predictions}), 200


if __name__ == '__main__':
    app.run(host=host, port=5000)

