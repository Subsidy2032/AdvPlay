import pandas as pd
import numpy as np
import os

from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader

class NPZDatasetLoader(BaseDatasetLoader, source_type="npz"):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self) -> np.ndarray:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"NPZ file not found: {self.path}")

        dataset_file = np.load(self.path)
        arrays = [dataset_file[key] for key in dataset_file.files]

        arrays = [arr.reshape(len(arr), -1) if arr.ndim > 1 else arr.reshape(-1, 1) for arr in arrays]

        dataset = np.hstack(arrays)

        return dataset

