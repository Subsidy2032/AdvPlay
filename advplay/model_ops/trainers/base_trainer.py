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

    def __init__(self, X_train, y_train, config: dict = None):
        self.X_train = X_train
        self.y_train = y_train
        self.config = config

        if not (isinstance(X_train, (np.ndarray, pd.DataFrame))):
            raise TypeError(f"Expected X_train to be a Pandas DataFrame or a Numpy array, got {type(X_train)}")
        if not isinstance(y_train, (np.ndarray, pd.Series, list)):
            raise TypeError(f"Expected y_train to be array-like, got {type(y_train)}")
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train length {len(X_train)} != y_train length {len(y_train)}")
        if len(X_train) == 0:
            raise ValueError("Train set is empty")
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"Expected config to be a dict, got {type(config)}")

    def train(self):
        raise NotImplementedError("Subclasses must implement the train method.")

