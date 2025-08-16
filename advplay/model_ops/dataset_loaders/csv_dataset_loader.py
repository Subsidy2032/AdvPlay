import pandas as pd
import os

from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader

class CSVDatasetLoader(BaseDatasetLoader, source_type="csv"):
    def __init__(self, path: str, label_column: str):
        super().__init__(path, label_column)

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"CSV file not found: {self.path}")
        return pd.read_csv(self.path)
