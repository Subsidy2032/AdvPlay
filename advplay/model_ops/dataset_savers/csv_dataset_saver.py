import numpy as np
import pandas as pd
from pathlib import Path

from advplay.model_ops.dataset_savers.base_dataset_saver import BaseDatasetSaver
from advplay.variables import dataset_formats

class CSVDatasetSaver(BaseDatasetSaver, source_type=dataset_formats.CSV):
    def save(self):
        target = self.path if self.path.suffix else self.path.with_suffix(".csv")
        columns = list(self.metadata.get("columns", [])) or None

        pd.DataFrame(self.data, columns=columns).to_csv(target, index=False)