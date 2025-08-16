import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):
    registry = {}

    def __init_subclass__(cls, source_type: str):
        key = source_type
        if key in BaseDatasetLoader.registry:
            raise ValueError(f"Loader already registered for source '{source_type}'")
        super().__init_subclass__()
        BaseDatasetLoader.registry[key] = cls

    def __init__(self, path, label_column: str):
        self.path = path
        self.label_column = label_column

    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    def split_dataset(self):
        dataset = self.load()

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"Expected DataFrame from loader, got {type(dataset)}")

        if self.label_column not in dataset.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataset")

        if len(dataset) == 0:
            raise ValueError("Loaded dataset is empty")

        X = dataset.loc[:, dataset.columns != self.label_column]
        y = dataset[self.label_column]

        return X, y
