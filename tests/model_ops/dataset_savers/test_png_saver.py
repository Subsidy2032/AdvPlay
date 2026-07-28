import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from advplay.ml.data.dataset_loaders.png_dataset_loader import PNGDatasetLoader
from advplay.ml.data.dataset_savers.png_dataset_saver import PNGDatasetSaver


def write_image(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_png_saver_directory_roundtrip(tmp_path):
    source = tmp_path / "images"
    originals = {
        "a.png": np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8),
        "b.png": np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8),
    }
    for name, array in originals.items():
        write_image(source / name, array)

    dataset = PNGDatasetLoader(str(source)).load()

    destination = tmp_path / "out"
    PNGDatasetSaver(dataset.data, dataset.metadata, Path(destination)).save()

    for name, array in originals.items():
        np.testing.assert_array_equal(np.asarray(Image.open(destination / name)), array)


def test_png_saver_keeps_nested_filenames(tmp_path):
    source = tmp_path / "images"
    write_image(source / "class_0" / "a.png", np.zeros((4, 4), dtype=np.uint8))
    write_image(source / "class_1" / "b.png", np.full((4, 4), 255, dtype=np.uint8))

    dataset = PNGDatasetLoader(str(source)).load()

    destination = tmp_path / "out"
    PNGDatasetSaver(dataset.data, dataset.metadata, Path(destination)).save()

    assert (destination / "class_0" / "a.png").exists()
    assert (destination / "class_1" / "b.png").exists()


def test_png_saver_numbers_files_when_sample_count_changes(tmp_path):
    source = tmp_path / "images"
    for index in range(3):
        write_image(source / f"{index}.png", np.full((4, 4), index, dtype=np.uint8))

    dataset = PNGDatasetLoader(str(source)).load()

    destination = tmp_path / "out"
    PNGDatasetSaver(dataset.data[:2], dataset.metadata, Path(destination)).save()

    assert sorted(path.name for path in destination.iterdir()) == ["images_0.png", "images_1.png"]


def test_png_saver_writes_single_file_when_path_has_suffix(tmp_path):
    pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
    data = pixels[np.newaxis, np.newaxis, :, :]

    destination = tmp_path / "single.png"
    PNGDatasetSaver(data, {"image_shape": (1, 4, 4)}, Path(destination)).save()

    assert destination.is_file()
    np.testing.assert_array_equal(np.asarray(Image.open(destination)), pixels)


def test_png_saver_rescales_unit_range_floats(tmp_path):
    data = np.array([[[[0.0, 0.5], [1.0, 0.25]]]], dtype=np.float32)

    destination = tmp_path / "out"
    PNGDatasetSaver(data, {"image_shape": (1, 2, 2), "dataset_name": "scaled"}, Path(destination)).save()

    saved = np.asarray(Image.open(destination / "scaled_0.png"))
    np.testing.assert_array_equal(saved, np.array([[0, 128], [255, 64]], dtype=np.uint8))


def test_png_saver_clips_out_of_range_pixels(tmp_path):
    data = np.array([[[[-30.0, 12.0], [300.0, 200.0]]]], dtype=np.float32)

    destination = tmp_path / "out"
    PNGDatasetSaver(data, {"image_shape": (1, 2, 2), "dataset_name": "clipped"}, Path(destination)).save()

    saved = np.asarray(Image.open(destination / "clipped_0.png"))
    np.testing.assert_array_equal(saved, np.array([[0, 12], [255, 200]], dtype=np.uint8))


def test_png_saver_infers_shape_from_flat_rows(tmp_path):
    rows = np.tile(np.arange(784, dtype=np.uint8), (2, 1))

    destination = tmp_path / "out"
    PNGDatasetSaver(rows, {"dataset_name": "flat"}, Path(destination)).save()

    saved = np.asarray(Image.open(destination / "flat_0.png"))
    assert saved.shape == (28, 28)
    np.testing.assert_array_equal(saved.ravel(), rows[0])


def test_png_saver_unknown_row_length_raises(tmp_path):
    rows = np.zeros((2, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="Cannot infer an image shape"):
        PNGDatasetSaver(rows, {}, Path(tmp_path / "out")).save()
