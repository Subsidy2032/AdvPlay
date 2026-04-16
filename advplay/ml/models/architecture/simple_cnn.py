import torch
import torch.nn as nn

from advplay.ml.models.architecture.registry import register_model

@register_model
class SimplePytorchCNN(nn.Module):
    def __init__(self, in_ch, num_classes, conv_ch, input_shape):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, conv_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # no dummy forward, no shape guessing
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
