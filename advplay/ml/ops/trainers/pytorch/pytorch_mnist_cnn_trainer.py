import numpy as np

from advplay.ml.ops.trainers.pytorch.pytorch_trainer import PyTorchTrainer
from advplay.ml.models.architecture.mnist_cnn import MnistCNN
from advplay.variables import available_models, available_frameworks

class PyTorchMnistCNNTrainer(PyTorchTrainer, framework=available_frameworks.PYTORCH,
                             model=available_models.MNIST_CNN):
    def __init__(self, X_train, y_train, config: dict = None):
        super().__init__(X_train, y_train, config)

        if self.config is None:
            self.config = {}

        training_config = self.config.get("training", self.config)
        self.in_channels = training_config.get("in_ch", training_config.get("in_channels", 1))
        self.num_classes = training_config.get("num_classes", len(np.unique(y_train)))
        self.conv_channels = training_config.get("conv_ch", training_config.get("conv_channels", 8))
        input_shape = training_config.get("input_shape")

        self.model = MnistCNN(self.in_channels, self.num_classes, self.conv_channels, input_shape)
