import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod

from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset

class BaseDatasetLoader(ABC):
    registry = {}

    def __init_subclass__(cls, source_type: str):
        key = source_type
        if key in BaseDatasetLoader.registry:
            raise ValueError(f"Loader already registered for source '{source_type}'")
        super().__init_subclass__()

        cls.source_type = source_type
        BaseDatasetLoader.registry[key] = cls

    def __init__(self, path):
        self.path = path
        self.dataset_name = os.path.splitext(os.path.basename(self.path))[0]

    @abstractmethod
    def load(self) -> LoadedDataset:
        raise NotImplementedError("Subclasses must implement the load method")
