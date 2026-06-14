import numpy as np

from advplay.ml.data.preprocessors.base_preprocessor import BasePreprocessor
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset


class PixelScaler(BasePreprocessor, name="pixel_scaler"):
    """Scale raw 0-255 pixel values into the [0, 1] range (no mean/std shift)."""

    PIXEL_SCALE = 255.0
    IMAGE_SIZE = 28 * 28

    def normalize(self, x):
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32, copy=False)
        return x / self.PIXEL_SCALE

    def apply(self, dataset: LoadedDataset) -> LoadedDataset:
        data = np.asarray(dataset.data)
        metadata = dataset.metadata or {}
        key_columns = metadata.get("key_columns") or {}

        image_keys = [cols for cols in key_columns.values() if len(cols) == self.IMAGE_SIZE]

        if image_keys:
            data = data.astype(np.float32, copy=True)
            for cols in image_keys:
                data[:, cols] = self.normalize(data[:, cols])
            return LoadedDataset(data=data, source_type=dataset.source_type, metadata=dataset.metadata)

        per_row = int(np.prod(data.shape[1:])) if data.ndim > 1 else data.size
        if per_row == self.IMAGE_SIZE:
            return LoadedDataset(data=self.normalize(data),
                                 source_type=dataset.source_type, metadata=dataset.metadata)

        return dataset
