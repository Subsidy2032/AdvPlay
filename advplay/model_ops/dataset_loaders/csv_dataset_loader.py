import pandas as pd
import numpy as np
import os

from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader

class CSVDatasetLoader(BaseDatasetLoader, source_type="csv"):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self) -> np.ndarray:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"CSV file not found: {self.path}")
        df = pd.read_csv(self.path)
        return df.to_numpy()
