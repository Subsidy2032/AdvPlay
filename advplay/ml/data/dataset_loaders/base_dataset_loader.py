import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod
from pathlib import Path

from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset

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
        # Path().stem rather than basename(): a directory passed with a trailing slash
        # ("png:my_images/") basenames to "", which collapses every such dataset into a
        # single shared output directory named "_perturbed".
        self.dataset_name = Path(self.path).stem

    @abstractmethod
    def load(self) -> LoadedDataset:
        raise NotImplementedError("Subclasses must implement the load method")
