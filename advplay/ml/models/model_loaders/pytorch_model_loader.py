import os
import torch
from art.estimators.classification import PyTorchClassifier

from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.variables import available_frameworks
from advplay.ml.models.architecture.registry import MODEL_REGISTRY

class TorchModelLoader(BaseModelLoader, framework=available_frameworks.PYTORCH):
    def __init__(self, model_path: str, model, config):
        super().__init__(model_path)
        self.model_class = MODEL_REGISTRY[model]
        self.config = config
        self.map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            model = self.model_class(**self.config.get("training"))
            model.load_state_dict(torch.load(self.model_path, map_location=self.map_location))
        except Exception as e:
            raise AttributeError(f"Failed loading model: {self.model_path}. Error: {e}")

        return model

    def load_art_classifier(self, loss, input_shape, nb_classes, clip_values):
        model = self.load()
        return PyTorchClassifier(model, loss=loss, input_shape=input_shape, nb_classes=nb_classes, clip_values=clip_values)
