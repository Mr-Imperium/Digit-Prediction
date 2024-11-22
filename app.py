from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import os
from model import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the model
from model import load_model
model = load_model("ml_with_pytorch_model.pth")
print("Model loaded successfully.")

# Preprocessing for images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess and predict
    input_tensor = preprocess_image(file_path)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_digit = torch.argmax(output).item()

    os.remove(file_path)  # Clean up after prediction
    return jsonify({'digit': predicted_digit})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
