import pandas as pd
import numpy as np
import os

from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.variables import dataset_formats

class CSVDatasetLoader(BaseDatasetLoader, source_type=dataset_formats.CSV):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self) -> LoadedDataset:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"CSV file not found: {self.path}")

        df = pd.read_csv(self.path)
        data = df.to_numpy()
        if df.shape[1] == 1:
            data = data.ravel()

        metadata = {"columns": df.columns}

        return LoadedDataset(data, source_type=self.source_type, metadata=metadata)
