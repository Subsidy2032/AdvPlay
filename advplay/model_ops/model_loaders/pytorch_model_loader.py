import os
import torch
from art.estimators.classification import PyTorchClassifier

from advplay.model_ops.model_loaders.base_model_loader import BaseModelLoader
from advplay.variables import available_frameworks

class TorchModelLoader(BaseModelLoader, framework=available_frameworks.PYTORCH):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            model = torch.load(self.model_path, map_location=self.map_location)
        except Exception as e:
            raise AttributeError(f"Failed loading model: {self.model_path}. Error: {e}")

        return model

    def load_art_classifier(self, loss, input_shape, nb_classes):
        model = self.load()
        return PyTorchClassifier(model, loss=loss, input_shape=input_shape, nb_classes=nb_classes)
