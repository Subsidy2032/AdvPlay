import numpy as np
import pandas as pd
import random
import os
import joblib
from scipy import sparse

from advplay.variables import available_frameworks
from advplay import paths

class BaseTrainer:
    registry = {}

    def __init_subclass__(cls, framework: str, model: str):
        key = (framework, model)
        if key in BaseTrainer.registry:
            raise ValueError(f"Subclass already registered for {framework} + {model}")

        super().__init_subclass__()
        BaseTrainer.registry[key] = cls

    def __init__(self, X_train, y_train, config: dict = None):
        self.X_train = X_train
        self.y_train = y_train
        self.config = config

        if not (isinstance(X_train, (np.ndarray, pd.DataFrame)) or sparse.issparse(X_train)):
            raise TypeError(f"Expected X_train to be a Pandas DataFrame, Numpy array, or sparse matrix, got {type(X_train)}")
        if not isinstance(y_train, (np.ndarray, pd.Series, list)):
            raise TypeError(f"Expected y_train to be array-like, got {type(y_train)}")
        n_x = X_train.shape[0]
        if n_x != len(y_train):
            raise ValueError(f"X_train length {n_x} != y_train length {len(y_train)}")
        if n_x == 0:
            raise ValueError("Train set is empty")
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"Expected config to be a dict, got {type(config)}")

    def train(self):
        raise NotImplementedError("Subclasses must implement the train method.")

