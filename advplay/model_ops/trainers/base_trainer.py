import numpy as np
import pandas as pd
import random

class BaseTrainer:
    registry = {}

    def __init_subclass__(cls, training_algorithm: str):
        if training_algorithm in BaseTrainer.registry.keys():
            raise ValueError(f"Two subclass are using the same training algorithm: {training_algorithm}")

        super().__init_subclass__()
        BaseTrainer.registry[training_algorithm] = cls

    def __init__(self, model_name, dataset, label_column: str, test_portion: float, seed: int = None):
        self.model_name = model_name
        self.dataset = dataset
        self.label_column = label_column
        self.test_portion = test_portion
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
