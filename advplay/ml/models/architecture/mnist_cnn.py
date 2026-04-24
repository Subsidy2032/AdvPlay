import torch
import torch.nn as nn
import torch.nn.functional as F

from advplay.ml.models.architecture.registry import register_model
from advplay.variables import available_models

@register_model(available_models.MNIST_CNN)
class MnistCNN(nn.Module):
    def __init__(self, in_ch, num_classes, conv_ch, input_shape):
        super(MnistCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3, padding=1)
        # Output: (Batch, 32, 28, 28)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (Batch, 32, 14, 14)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        # Output: (Batch, 64, 14, 14)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (Batch, 64, 7, 7)

        self._feature_size = 64 * 7 * 7  # 3136
        self.fc1 = nn.Linear(self._feature_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        self.register_buffer("pixel_scale", torch.tensor(255.0), persistent=False)
        self.register_buffer("mean", torch.tensor(0.1307).view(1, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(0.3081).view(1, 1, 1, 1), persistent=False)

    def forward(self, x):
        x = x / self.pixel_scale
        x = (x - self.mean) / self.std
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self._feature_size)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
