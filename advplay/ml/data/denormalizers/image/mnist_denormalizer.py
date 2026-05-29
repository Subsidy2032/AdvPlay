import numpy as np

from advplay.ml.data.denormalizers.base_denormalizer import BaseDenormalizer
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset


class MnistDenormalizer(BaseDenormalizer, name="mnist_normalizer"):
    PIXEL_SCALE = 255.0
    MEAN = 0.1307
    STD = 0.3081
    IMAGE_SIZE = 28 * 28

    def denormalize(self, x):
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32, copy=False)
        return (x * self.STD + self.MEAN) * self.PIXEL_SCALE

    def apply(self, dataset: LoadedDataset) -> LoadedDataset:
        data = np.asarray(dataset.data)
        metadata = dataset.metadata or {}
        key_columns = metadata.get("key_columns") or {}

        image_keys = [cols for cols in key_columns.values() if len(cols) == self.IMAGE_SIZE]

        if image_keys:
            data = data.astype(np.float32, copy=True)
            for cols in image_keys:
                data[:, cols] = self.denormalize(data[:, cols])
            return LoadedDataset(data=data, source_type=dataset.source_type, metadata=dataset.metadata)

        per_row = int(np.prod(data.shape[1:])) if data.ndim > 1 else data.size
        if per_row == self.IMAGE_SIZE:
            return LoadedDataset(data=self.denormalize(data),
                                 source_type=dataset.source_type, metadata=dataset.metadata)

        return dataset
