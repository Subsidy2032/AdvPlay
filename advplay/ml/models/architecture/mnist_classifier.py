import torch.nn as nn
import torch.nn.functional as F
import torch

from advplay.ml.models.architecture.registry import register_model
from advplay.variables import available_models

@register_model(available_models.MNIST_CLASSIFIER)
class MNISTClassifier(nn.Module):
    """LeNet-5 style classifier.

    Architecture:
    - Conv1: 1 -> 6 channels, 5x5 kernel, Tanh activation, 2x2 avg pooling
    - Conv2: 6 -> 16 channels, 5x5 kernel, Tanh activation, 2x2 avg pooling
    - FC1: 256 -> 120, Tanh activation
    - FC2: 120 -> 84, Tanh activation
    - FC3: 84 -> 10, log-softmax output
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning log-softmax outputs.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28) in [0,1] range

        Returns:
            Log-softmax output of shape (batch, 10)
        """
        x = self.act(self.conv1(x))    # (B,6,24,24)
        x = self.pool(x)               # (B,6,12,12)
        x = self.act(self.conv2(x))    # (B,16,8,8)
        x = self.pool(x)               # (B,16,4,4)
        x = torch.flatten(x, 1)        # (B,256)
        x = self.act(self.fc1(x))      # (B,120)
        x = self.act(self.fc2(x))      # (B,84)
        x = self.fc3(x)                # (B,10)
        return F.log_softmax(x, dim=1)
