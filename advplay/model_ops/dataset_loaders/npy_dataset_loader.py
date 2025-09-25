import numpy as np
import os

from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.variables import dataset_formats

class NPYDatasetLoader(BaseDatasetLoader, source_type=dataset_formats.NPY):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self) -> LoadedDataset:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"NPY file not found: {self.path}")

        arr = np.load(self.path)
        arr = arr.reshape(len(arr), -1) if arr.ndim > 1 else arr.reshape(-1, 1)

        return LoadedDataset(arr, source_type=self.source_type)
