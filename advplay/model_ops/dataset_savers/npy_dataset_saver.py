import numpy as np
from pathlib import Path

from advplay.model_ops.dataset_savers.base_dataset_saver import BaseDatasetSaver
from advplay.variables import dataset_formats

class NPYDatasetSaver(BaseDatasetSaver, source_type=dataset_formats.NPY):
    def save(self):
        target = self.path if self.path.suffix else self.path.with_suffix(".npy")
        np.save(str(target), self.data)
