import numpy as np
import pytest

from advplay.ml.data.dataset_loaders.npz_dataset_loader import NPZDatasetLoader


def test_npz_uniform_row_counts(tmp_path):
    path = tmp_path / "uniform.npz"
    features = np.arange(20).reshape(10, 2)
    labels = np.array([0, 1] * 5)
    np.savez(path, features=features, labels=labels)

    dataset = NPZDatasetLoader(str(path)).load()

    assert dataset.data.shape == (10, 3)
    assert dataset.metadata["keys"] == ["features", "labels"]
    assert dataset.metadata["key_columns"] == {"features": [0, 1], "labels": [2]}
    np.testing.assert_array_equal(dataset.data[:, :2], features)
    np.testing.assert_array_equal(dataset.data[:, 2], labels)


def test_npz_train_test_split_row_counts(tmp_path):
    path = tmp_path / "split.npz"
    Xtr = np.arange(14).reshape(7, 2).astype(float)
    ytr = np.array([1, 0, 1, 0, 1, 0, 1])
    Xte = np.arange(6).reshape(3, 2).astype(float) + 100
    yte = np.array([0, 1, 0])
    np.savez(path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    dataset = NPZDatasetLoader(str(path)).load()

    assert dataset.data.shape == (10, 3)
    np.testing.assert_array_equal(dataset.data[:7, :2], Xtr)
    np.testing.assert_array_equal(dataset.data[:7, 2], ytr)
    np.testing.assert_array_equal(dataset.data[7:, :2], Xte)
    np.testing.assert_array_equal(dataset.data[7:, 2], yte)
    assert dataset.metadata["key_columns"]["ytr"] == [2]
    assert dataset.metadata["key_columns"]["yte"] == [2]


def test_npz_loads_label_flipping_dataset():
    dataset = NPZDatasetLoader(
        "/home/amulet/projects/AdvPlay/resources/datasets/label_flipping_dataset.npz"
    ).load()

    assert dataset.data.shape == (1000, 3)
    assert dataset.metadata["key_columns"]["ytr"] == [2]
    assert dataset.metadata["key_columns"]["yte"] == [2]


def test_npz_incompatible_groups_raises(tmp_path):
    path = tmp_path / "bad.npz"
    a = np.arange(10).reshape(5, 2)
    b = np.arange(6).reshape(3, 2)
    c = np.arange(3)
    np.savez(path, a=a, b=b, c=c)

    with pytest.raises(ValueError, match="cannot be combined"):
        NPZDatasetLoader(str(path)).load()


def test_npz_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="NPZ file not found"):
        NPZDatasetLoader(str(tmp_path / "missing.npz")).load()
