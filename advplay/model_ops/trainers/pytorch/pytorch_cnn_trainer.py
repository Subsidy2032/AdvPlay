import torch
import torch.nn as nn
import numpy as np
import math

from advplay.model_ops.trainers.pytorch.pytorch_trainer import PyTorchTrainer
from advplay.variables import available_models, available_frameworks

class PyTorchCNNTrainer(PyTorchTrainer, framework=available_frameworks.PYTORCH,
                        model=available_models.CNN):
    def __init__(self, X_train, y_train, config: dict = None):
        super().__init__(X_train, y_train, config)

        if self.config is None:
            self.config = {}

        self.in_channels = self.config.get("in_channels", 1)
        self.num_classes = self.config.get("num_classes", len(np.unique(y_train)))
        self.conv_channels = self.config.get("conv_channels", 8)
        self.channels_first = self.config.get("channels_first", True)
        self.height = self.config.get("height")
        self.width = self.config.get("width")

        input_shape = (self.height, self.width)
        self.model = SimpleCNN(self.in_channels, self.num_classes, self.conv_channels, input_shape)

class SimpleCNN(nn.Module):
    def __init__(self, in_ch, num_classes, conv_ch, input_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, conv_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, *input_shape)
            dummy_out = self.pool(torch.relu(self.conv(dummy)))
            flattened_size = dummy_out.view(1, -1).size(1)
        self.fc = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @staticmethod
    def preprocess_data(X, height=None, width=None, channels_first=True, in_channels=1):
        X_tensor = torch.tensor(X, dtype=torch.float32)

        if X_tensor.ndim == 2:
            if height is None and width is None:
                height = int(math.sqrt(X_tensor.shape[1]))
                width = height

            X_tensor = X_tensor.view(-1, in_channels, height, width)

        if X_tensor.ndim == 3:
            X_tensor = X_tensor.unsqueeze(1)

        if not channels_first and X_tensor.ndim == 4:
            X_tensor = X_tensor.permute(0, 3, 1, 2)

        return X_tensor

