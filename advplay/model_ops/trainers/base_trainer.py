import numpy as np
import pandas as pd
import random
import os
import joblib

from advplay.variables import available_frameworks
from advplay import paths

class BaseTrainer:
    registry = {}

    def __init_subclass__(cls, framework: str, training_algorithm: str):
        key = (framework, training_algorithm)
        if key in BaseTrainer.registry:
            raise ValueError(f"Subclass already registered for {framework} + {training_algorithm}")

        super().__init_subclass__()
        BaseTrainer.registry[key] = cls

    def __init__(self, model_name, dataset, label_column: str, test_portion: float, config: dict = None, seed: int = None):
        self.model_name = model_name
        self.dataset = dataset
        self.label_column = label_column
        self.test_portion = test_portion
        self.config = config
        self.seed = seed

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"Expected dataset to be a Pandas DataFrame, got {type(dataset)}")
        if label_column not in dataset.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        if not (0 < test_portion < 1):
            raise ValueError("Test portion must be between 0 and 1")

        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        raise NotImplementedError("Subclasses must implement the train method.")

    def save_model(self, model):
        raise NotImplementedError("Subclasses must implement the save_model method")

    def save_sklearn_model(self, model):
        file_path = paths.MODELS / available_frameworks.SKLEARN / f"{self.model_name}.joblib"
        os.makedirs(file_path.parent, exist_ok=True)

        joblib.dump(model, file_path)

        print(f"Model saved to {file_path}")
        return file_path
