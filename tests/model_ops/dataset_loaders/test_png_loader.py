import numpy as np
import pytest
from PIL import Image

from advplay.ml.data.dataset_loaders.png_dataset_loader import PNGDatasetLoader


def write_image(path, array, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


def test_png_loads_single_grayscale_file(tmp_path):
    pixels = np.arange(28 * 28, dtype=np.uint8).reshape(28, 28)
    path = tmp_path / "digit.png"
    write_image(path, pixels)

    dataset = PNGDatasetLoader(str(path)).load()

    assert dataset.data.shape == (1, 1, 28, 28)
    assert dataset.data.dtype == np.uint8
    assert dataset.metadata["image_shape"] == (1, 28, 28)
    assert dataset.metadata["mode"] == "L"
    assert dataset.metadata["filenames"] == ["digit.png"]
    assert dataset.metadata["dataset_name"] == "digit"
    np.testing.assert_array_equal(dataset.data[0, 0], pixels)


def test_png_loads_directory_channels_first_and_sorted(tmp_path):
    images = {
        "b.png": np.full((4, 4, 3), 20, dtype=np.uint8),
        "a.png": np.full((4, 4, 3), 10, dtype=np.uint8),
        "c.png": np.full((4, 4, 3), 30, dtype=np.uint8),
    }
    for name, array in images.items():
        write_image(tmp_path / name, array)

    dataset = PNGDatasetLoader(str(tmp_path)).load()

    assert dataset.data.shape == (3, 3, 4, 4)
    assert dataset.metadata["filenames"] == ["a.png", "b.png", "c.png"]
    assert dataset.metadata["mode"] == "RGB"
    np.testing.assert_array_equal(dataset.data[:, 0, 0, 0], [10, 20, 30])


def test_png_loads_nested_directories(tmp_path):
    write_image(tmp_path / "one" / "a.png", np.zeros((2, 2), dtype=np.uint8))
    write_image(tmp_path / "two" / "b.png", np.full((2, 2), 255, dtype=np.uint8))
    (tmp_path / "notes.txt").write_text("ignored")

    dataset = PNGDatasetLoader(str(tmp_path)).load()

    assert dataset.data.shape == (2, 1, 2, 2)
    assert dataset.metadata["filenames"] == ["one/a.png", "two/b.png"]


def test_png_expands_palette_images(tmp_path):
    palette = Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).convert("P")
    path = tmp_path / "palette.png"
    palette.save(path)

    dataset = PNGDatasetLoader(str(path)).load()

    assert dataset.data.shape == (1, 3, 3, 3)
    assert dataset.metadata["mode"] == "RGB"


def test_png_mismatched_shapes_raises(tmp_path):
    write_image(tmp_path / "small.png", np.zeros((2, 2), dtype=np.uint8))
    write_image(tmp_path / "large.png", np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="cannot be combined"):
        PNGDatasetLoader(str(tmp_path)).load()


def test_png_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="No PNG files found"):
        PNGDatasetLoader(str(empty)).load()


def test_png_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="PNG file not found"):
        PNGDatasetLoader(str(tmp_path / "missing.png")).load()
