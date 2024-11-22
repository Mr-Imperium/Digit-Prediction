from flask import Flask, request, jsonify, render_template
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io

# Load your trained model
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, image_tensor):
        image_tensor = self.flatten(image_tensor)
        logits = self.linear_relu_stack(image_tensor)
        return logits

# Initialize the model
input_size = 28 * 28
hidden_size = 512
num_classes = 10
device = torch.device("cpu")

model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

app = Flask(__name__)

# Define a prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image = request.files["image"]
    image = Image.open(io.BytesIO(image.read())).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))
    tensor = ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(1).item()
    return jsonify({"prediction": prediction})

# Define a home route for your front-end
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
