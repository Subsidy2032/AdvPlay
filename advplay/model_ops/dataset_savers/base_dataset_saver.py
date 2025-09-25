import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod
from pathlib import Path

from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset

class BaseDatasetSaver(ABC):
    registry = {}

    def __init_subclass__(cls, source_type: str):
        key = source_type
        if key in BaseDatasetSaver.registry:
            raise ValueError(f"Saver already registered for source '{source_type}'")
        super().__init_subclass__()

        cls.source_type = source_type
        BaseDatasetSaver.registry[key] = cls

    def __init__(self, data: np.ndarray, metadata: dict, path: Path):
        self.data = data
        self.metadata = metadata
        self.path = path

    @abstractmethod
    def save(self):
        raise NotImplementedError("Subclasses must implement the save method")
