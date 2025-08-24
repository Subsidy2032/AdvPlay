import pandas as pd
import numpy as np
import os

from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader

class NPZDatasetLoader(BaseDatasetLoader, source_type="npz"):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"NPZ file not found: {self.path}")

        dataset_file = np.load(self.path)

        for key in dataset_file.files:
            if dataset_file[key].ndim > 2:
                raise AttributeError(f"Array '{key}' has {dataset_file[key].ndim} dimensions; only 1D/2D supported")

        if 'X' in dataset_file.files and 'y' in dataset_file.files:
                X = dataset_file['X'].reshape(len(dataset_file['y']), -1)
                df = pd.DataFrame(X)
                df['y'] = dataset_file['y']
                return df

        return pd.DataFrame({key: dataset_file[key] for key in dataset_file.files})
