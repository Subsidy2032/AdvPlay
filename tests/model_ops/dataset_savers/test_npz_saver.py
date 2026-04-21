import numpy as np
from pathlib import Path

from advplay.ml.data.dataset_loaders.npz_dataset_loader import NPZDatasetLoader
from advplay.ml.data.dataset_savers.npz_dataset_saver import NPZDatasetSaver


def test_npz_saver_uniform_roundtrip(tmp_path):
    src = tmp_path / "uniform.npz"
    features = np.arange(20).reshape(10, 2)
    labels = np.array([0, 1] * 5)
    np.savez(src, features=features, labels=labels)

    dataset = NPZDatasetLoader(str(src)).load()

    dst = tmp_path / "uniform_out.npz"
    NPZDatasetSaver(dataset.data, dataset.metadata, Path(dst)).save()

    loaded = np.load(dst)
    np.testing.assert_array_equal(loaded["features"], features)
    np.testing.assert_array_equal(loaded["labels"], labels)


def test_npz_saver_train_test_split_roundtrip(tmp_path):
    src = tmp_path / "split.npz"
    Xtr = np.arange(14).reshape(7, 2).astype(float)
    ytr = np.array([1, 0, 1, 0, 1, 0, 1])
    Xte = np.arange(6).reshape(3, 2).astype(float) + 100
    yte = np.array([0, 1, 0])
    np.savez(src, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    dataset = NPZDatasetLoader(str(src)).load()

    dst = tmp_path / "split_out.npz"
    NPZDatasetSaver(dataset.data, dataset.metadata, Path(dst)).save()

    loaded = np.load(dst)
    np.testing.assert_array_equal(loaded["Xtr"], Xtr)
    np.testing.assert_array_equal(loaded["ytr"], ytr)
    np.testing.assert_array_equal(loaded["Xte"], Xte)
    np.testing.assert_array_equal(loaded["yte"], yte)
    assert loaded["Xtr"].shape == Xtr.shape
    assert loaded["ytr"].shape == ytr.shape


def test_npz_saver_label_flipping_dataset_roundtrip(tmp_path):
    src = "/home/amulet/projects/AdvPlay/resources/datasets/label_flipping_dataset.npz"
    dataset = NPZDatasetLoader(src).load()

    dst = tmp_path / "label_flipping_out.npz"
    NPZDatasetSaver(dataset.data, dataset.metadata, Path(dst)).save()

    original = np.load(src)
    roundtrip = np.load(dst)
    for key in original.files:
        np.testing.assert_array_equal(roundtrip[key], original[key])
        assert roundtrip[key].shape == original[key].shape


def test_npz_saver_handles_row_count_mismatch(tmp_path):
    src = tmp_path / "split.npz"
    Xtr = np.arange(14).reshape(7, 2).astype(float)
    ytr = np.array([1, 0, 1, 0, 1, 0, 1])
    Xte = np.arange(6).reshape(3, 2).astype(float) + 100
    yte = np.array([0, 1, 0])
    np.savez(src, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    dataset = NPZDatasetLoader(str(src)).load()

    subset = dataset.data[:8]

    dst = tmp_path / "subset_out.npz"
    NPZDatasetSaver(subset, dataset.metadata, Path(dst)).save()

    loaded = np.load(dst)
    assert set(loaded.files) == {"Xtr", "ytr"}
    assert loaded["Xtr"].shape == (8, 2)
    assert loaded["ytr"].shape == (8,)
    np.testing.assert_array_equal(loaded["Xtr"], subset[:, :2])
    np.testing.assert_array_equal(loaded["ytr"], subset[:, 2])


def test_npz_saver_adds_suffix_when_missing(tmp_path):
    src = tmp_path / "uniform.npz"
    features = np.arange(4).reshape(2, 2)
    labels = np.array([0, 1])
    np.savez(src, features=features, labels=labels)

    dataset = NPZDatasetLoader(str(src)).load()

    dst_no_suffix = tmp_path / "out_no_suffix"
    NPZDatasetSaver(dataset.data, dataset.metadata, Path(dst_no_suffix)).save()

    assert (tmp_path / "out_no_suffix.npz").exists()
