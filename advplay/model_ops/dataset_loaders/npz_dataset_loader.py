import pandas as pd
import numpy as np
import os

from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.variables import dataset_formats

class NPZDatasetLoader(BaseDatasetLoader, source_type=dataset_formats.NPZ):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self) -> LoadedDataset:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"NPZ file not found: {self.path}")

        dataset_file = np.load(self.path)

        keys = list(dataset_file.files)
        shapes = {k: dataset_file[k].shape for k in keys}

        arrays = [dataset_file[key] for key in dataset_file.files]
        arrays = [arr.reshape(len(arr), -1) if arr.ndim > 1 else arr.reshape(-1, 1) for arr in arrays]

        dataset = np.hstack(arrays)
        metadata = {"keys": keys, "shapes": shapes}

        return LoadedDataset(dataset, source_type=self.source_type, metadata=metadata)
