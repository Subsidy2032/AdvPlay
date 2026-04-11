import numpy as np

from advplay.ml.ops.trainers.pytorch.pytorch_trainer import PyTorchTrainer
from advplay.ml.models.architecture.simple_cnn import SimplePytorchCNN
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
        self.model = SimplePytorchCNN(self.in_channels, self.num_classes, self.conv_channels, input_shape)
