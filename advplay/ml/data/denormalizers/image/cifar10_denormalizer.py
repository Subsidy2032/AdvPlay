import numpy as np

from advplay.ml.data.denormalizers.base_denormalizer import BaseDenormalizer
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset


class Cifar10Denormalizer(BaseDenormalizer, name="cifar10_normalizer"):
    # Inverse of Cifar10Normalizer: maps normalized values back to [0, 1] pixel space.
    PIXEL_SCALE = 1.0
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.247, 0.2435, 0.2616]
    CHANNELS = 3
    SIDE = 32
    IMAGE_SIZE = CHANNELS * SIDE * SIDE  # 3 * 32 * 32 = 3072

    def denormalize(self, x):
        # x is channels-first (..., C, H, W); mean/std broadcast over each channel.
        x = np.asarray(x, dtype=np.float32)
        mean = np.asarray(self.MEAN, dtype=np.float32).reshape(self.CHANNELS, 1, 1)
        std = np.asarray(self.STD, dtype=np.float32).reshape(self.CHANNELS, 1, 1)
        return (x * std + mean) * self.PIXEL_SCALE

    def _denormalize_block(self, block):
        # Reshape a flat row block into channels-first images, denormalize, flatten back.
        images = block.reshape(-1, self.CHANNELS, self.SIDE, self.SIDE)
        return self.denormalize(images).reshape(block.shape)

    def apply(self, dataset: LoadedDataset) -> LoadedDataset:
        data = np.asarray(dataset.data)
        metadata = dataset.metadata or {}
        key_columns = metadata.get("key_columns") or {}

        image_keys = [cols for cols in key_columns.values() if len(cols) == self.IMAGE_SIZE]

        if image_keys:
            data = data.astype(np.float32, copy=True)
            for cols in image_keys:
                data[:, cols] = self._denormalize_block(data[:, cols])
            return LoadedDataset(data=data, source_type=dataset.source_type, metadata=dataset.metadata)

        per_row = int(np.prod(data.shape[1:])) if data.ndim > 1 else data.size
        if per_row == self.IMAGE_SIZE:
            denormalized = self.denormalize(data.reshape(-1, self.CHANNELS, self.SIDE, self.SIDE)).reshape(data.shape)
            return LoadedDataset(data=denormalized,
                                 source_type=dataset.source_type, metadata=dataset.metadata)

        return dataset
