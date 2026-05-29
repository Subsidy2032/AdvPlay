import torch.nn as nn
import torch.nn.functional as F
import torch

from advplay.ml.models.architecture.registry import register_model
from advplay.variables import available_models

@register_model(available_models.SIMPLE_CLASSIFIER)
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        """Forward pass with internal normalization.

        Args:
            x01: Input tensor in [0,1] with shape (N, 1, 28, 28)

        Returns:
            Log-probabilities with shape (N, 10)
        """
        x = x01
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)
