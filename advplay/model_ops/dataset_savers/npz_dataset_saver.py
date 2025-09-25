import numpy as np
from pathlib import Path

from advplay.model_ops.dataset_savers.base_dataset_saver import BaseDatasetSaver
from advplay.variables import dataset_formats

class NPZDatasetSaver(BaseDatasetSaver, source_type=dataset_formats.NPZ):
    def save(self):
        target = self.path if self.path.suffix else self.path.with_suffix(".npz")
        keys, shapes = self.metadata["keys"], self.metadata["shapes"]
        arrays, offset = {}, 0

        for key in keys:
            shape = tuple(shapes[key])
            cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            slice_ = self.data[:, offset:offset + cols]
            offset += cols
            arrays[key] = slice_.reshape(shape)

        np.savez(str(target), **arrays)
