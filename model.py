import torch
import torch.nn as nn

# Define the model class (match the structure used during training)
class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),  # Input size for 28x28 images
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # Output size for 10 digits (0-9)
        )

    def forward(self, x):
        return self.fc(x)

# Function to load the model
def load_model(model_path):
    model = DigitModel()  # Initialize the model structure
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load weights
    return model
