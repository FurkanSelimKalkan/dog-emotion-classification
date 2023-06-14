from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import ImageClassifier


# Load the first trained model
model_first = ImageClassifier(num_classes=4)
model_first.load_state_dict(torch.load("cleanedDataAlexPre8-05.pth", map_location=torch.device('cuda')))
#model_first.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cuda')))
model_first.eval()  # Set the model to evaluation mode

# Define the class labels and their corresponding custom output names
class_labels = ["angry", "happy", "relaxed", "sad"]
output_names = ["angry", "happy", "relaxed", "sad"]

# Preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

@app.route('/predict', methods=['POST'])
#@cross_origin()
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image = request.files['image']
    print(image)
    image = Image.open(image)
    input_image = preprocess(image)
    input_image = input_image.unsqueeze(0)  # Add a batch dimension
    with torch.no_grad():
        output_first = model_first(input_image)
        _, predicted_class_first = torch.max(output_first, 1)
        predicted_label_first = class_labels[predicted_class_first.item()]
        output_name_first = output_names[predicted_class_first.item()]

    return jsonify({'prediction': output_name_first}), 200


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

        with torch.no_grad():
            output_first = model_first(input_image)
            _, predicted_class_first = torch.max(output_first, 1)
            predicted_label_first = class_labels[predicted_class_first.item()]
            output_name_first = output_names[predicted_class_first.item()]

        predictions.append(output_name_first)

    return jsonify({'predictions': predictions}), 200


if __name__ == '__main__':

    app.run()
