import numpy as np
from pathlib import Path
from PIL import Image

from advplay.ml.data.dataset_loaders.base_dataset_loader import BaseDatasetLoader
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.variables import dataset_formats

class PNGDatasetLoader(BaseDatasetLoader, source_type=dataset_formats.PNG):
    """Load a single PNG file, or every PNG under a directory, as one image batch.

    Images come back as raw 0-255 pixels shaped (samples, channels, height, width) - the
    channels-first layout the preprocessors and model architectures expect. Point --samples
    at a directory with 'png:my_images' so the loader is picked by prefix instead of by
    file extension.
    """

    EXTENSION = ".png"

    def __init__(self, path: str):
        super().__init__(path)

    def load(self) -> LoadedDataset:
        root = Path(self.path)
        if not root.exists():
            raise FileNotFoundError(f"PNG file not found: {self.path}")

        files = self.collect_files(root)
        if not files:
            raise FileNotFoundError(f"No PNG files found in directory: {self.path}")

        images, modes = [], []
        for file in files:
            image, mode = self.read_image(file)
            images.append(image)
            modes.append(mode)

        shapes = {image.shape for image in images}
        if len(shapes) > 1:
            raise ValueError(
                f"PNG images cannot be combined: found shapes {sorted(shapes)} in {self.path}. "
                f"All images in a dataset must share the same size and channel count"
            )

        data = np.stack(images)

        metadata = {
            "filenames": [str(file.relative_to(root)) if root.is_dir() else file.name for file in files],
            "image_shape": tuple(data.shape[1:]),
            "mode": modes[0],
            "dataset_name": self.dataset_name,
            "dataset_path": self.path,
        }

        return LoadedDataset(data, source_type=self.source_type, metadata=metadata)

    def collect_files(self, root: Path):
        if not root.is_dir():
            return [root]

        return sorted(path for path in root.rglob("*")
                      if path.is_file() and path.suffix.lower() == self.EXTENSION)

    def read_image(self, file: Path):
        with Image.open(file) as image:
            # Palette images decode to colour indices, so expand them into real channels first.
            if image.mode == "P":
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            mode = image.mode
            array = np.asarray(image, dtype=np.uint8)

        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))

        return array, mode
