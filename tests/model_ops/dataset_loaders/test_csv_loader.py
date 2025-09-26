# test_dataset_loaders.py
import os
import pandas as pd
import pytest
from pathlib import Path
from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader
from advplay.model_ops.dataset_loaders.csv_dataset_loader import CSVDatasetLoader
from advplay.model_ops.registry import load_dataset
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset

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
    return str(file_path)

@pytest.fixture
def source(temp_csv):
    return os.path.splitext(temp_csv)[1][1:]

# --- Tests ---
def test_csv_loader_loads(temp_csv):
    loader = CSVDatasetLoader(temp_csv)
    loaded_dataset = loader.load()
    assert isinstance(loaded_dataset, LoadedDataset)
    assert "label" in list(loaded_dataset.metadata.get("columns"))
    assert len(loaded_dataset.data) == 3

def test_registry_load_dataset(source, temp_csv):
    loaded_dataset = load_dataset(source, temp_csv)
    assert isinstance(loaded_dataset, LoadedDataset)
    assert "label" in list(loaded_dataset.metadata.get("columns"))
    assert len(loaded_dataset.data) == 3

def test_file_not_found(source, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(source, tmp_path / "nonexistent.csv")