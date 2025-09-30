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

        metadata = {"dataset_name": self.dataset_name, "dataset_path": self.path}
        return LoadedDataset(arr, source_type=self.source_type, metadata=metadata)
