import torch
import torch.nn as nn

# Define your model class (same structure as in training)
class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

def load_model(model_path):
    model = DigitModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model
