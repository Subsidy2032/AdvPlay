import torch
import torch.nn as nn

from advplay.ml.models.architecture.registry import register_model
from advplay.variables import available_models

@register_model(available_models.MNIST_CNN)
class MnistCNN(nn.Module):
    def __init__(self, in_ch, num_classes, conv_ch, input_shape):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, conv_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = torch.clamp(x, 0.0, 1.0)
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
