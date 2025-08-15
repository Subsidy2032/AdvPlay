import os
import joblib

from advplay.model_ops.loaders.base_loader import BaseLoader
from advplay.variables import available_frameworks

class SklearnLoader(BaseLoader, framework=available_frameworks.SKLEARN):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            model = joblib.load(self.model_path)

        except Exception as e:
            raise AttributeError(f"Failed loading model: {self.model_path}")

        return model
