import numpy as np
from math import isqrt
from pathlib import Path
from PIL import Image

from advplay.ml.data.dataset_savers.base_dataset_saver import BaseDatasetSaver
from advplay.variables import dataset_formats

class PNGDatasetSaver(BaseDatasetSaver, source_type=dataset_formats.PNG):
    """Write an image batch back out as PNG files, one file per sample.

    The path is treated as a directory unless it ends in '.png' and a single image is being
    saved. Original filenames are reused when the metadata carries them and the sample count
    still matches, otherwise files are numbered '<dataset_name>_000.png'.
    """

    EXTENSION = ".png"
    CHANNEL_COUNTS = (1, 3, 4)
    PIXEL_SCALE = 255.0

    def save(self):
        path = Path(self.path)
        images = self.to_images(np.asarray(self.data))
        targets = self.resolve_targets(path, len(images))

        for image, target in zip(images, targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(target)

    def to_images(self, data):
        batch = self.to_batch(self.to_pixels(data))
        return [self.to_channels_last(image) for image in batch]

    def to_batch(self, data):
        """Reshape whatever the pipeline produced into channels-first (samples, C, H, W)."""
        image_shape = self.metadata.get("image_shape")
        if image_shape:
            return data.reshape((-1,) + tuple(image_shape))

        if data.ndim == 4:
            return data
        if data.ndim == 3:
            # Ambiguous without metadata: (C, H, W) if the first axis looks like channels,
            # otherwise a batch of grayscale images.
            return data[np.newaxis, ...] if data.shape[0] in self.CHANNEL_COUNTS else data[:, np.newaxis, ...]
        if data.ndim == 2:
            return data.reshape((-1,) + self.infer_image_shape(data.shape[1]))

        raise ValueError(
            f"Cannot save {data.ndim}-dimensional data as PNG. Provide an 'image_shape' "
            f"(channels, height, width) entry in the dataset metadata"
        )

    def infer_image_shape(self, features):
        """Guess the image shape of flat rows, assuming square images (784 -> (1, 28, 28))."""
        for channels in self.CHANNEL_COUNTS:
            if features % channels:
                continue
            side = isqrt(features // channels)
            if side * side == features // channels:
                return channels, side, side

        raise ValueError(
            f"Cannot infer an image shape for rows of {features} values. Provide an "
            f"'image_shape' (channels, height, width) entry in the dataset metadata"
        )

    def to_pixels(self, data):
        if np.issubdtype(data.dtype, np.floating):
            # Denormalizers hand back 0-255 floats, but data still in [0, 1] would otherwise
            # be truncated into an almost entirely black image.
            if data.size and np.nanmax(np.abs(data)) <= 1.0:
                data = data * self.PIXEL_SCALE
            # Round rather than truncate, so denormalized pixels land back on their original value.
            data = np.rint(np.clip(data, 0, self.PIXEL_SCALE))

        return data.astype(np.uint8)

    def to_channels_last(self, image):
        if image.shape[0] in self.CHANNEL_COUNTS:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = image[..., 0]

        return image

    def resolve_targets(self, path, count):
        if path.suffix.lower() == self.EXTENSION:
            if count == 1:
                return [path]
            # A single filename cannot hold a batch, so use a directory of that name instead.
            path = path.with_suffix("")

        filenames = list(self.metadata.get("filenames") or [])
        if len(filenames) != count:
            dataset_name = self.metadata.get("dataset_name") or path.name
            width = len(str(count - 1)) if count > 1 else 1
            filenames = [f"{dataset_name}_{index:0{width}d}{self.EXTENSION}" for index in range(count)]

        return [path / filename for filename in filenames]
