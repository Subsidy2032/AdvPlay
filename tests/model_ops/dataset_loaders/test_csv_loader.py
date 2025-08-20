# test_dataset_loaders.py
import os
import pandas as pd
import pytest
from pathlib import Path
from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader
from advplay.model_ops.dataset_loaders.csv_dataset_loader import CSVDatasetLoader
from advplay.model_ops.registry import load_dataset

# --- Fixtures ---
@pytest.fixture
def temp_csv(tmp_path):
    data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "label": [0, 1, 0]
    })
    file_path = tmp_path / "test.csv"
    data.to_csv(file_path, index=False)
    return file_path

# --- Tests ---
def test_csv_loader_loads(temp_csv):
    loader = CSVDatasetLoader(temp_csv, label_column="label")
    df = loader.load()
    assert isinstance(df, pd.DataFrame)
    assert "label" in df.columns
    assert len(df) == 3

def test_csv_loader_split_dataset(temp_csv):
    loader = CSVDatasetLoader(temp_csv, label_column="label")
    X, y = loader.split_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == ["feature1", "feature2"]
    assert y.tolist() == [0, 1, 0]

def test_registry_load_dataset(temp_csv):
    # Using registry loader function
    df = load_dataset("csv", temp_csv, label_column="label")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["feature1", "feature2", "label"]

def test_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVDatasetLoader(tmp_path / "nonexistent.csv", label_column="label").load()

def test_invalid_label_column(temp_csv):
    loader = CSVDatasetLoader(temp_csv, label_column="nonexistent")
    with pytest.raises(ValueError):
        loader.split_dataset()
