import pytest
import joblib
import tempfile
import os

from advplay.model_ops.model_loaders.sklearn_model_loader import SklearnModelLoader
from advplay.variables import available_frameworks
from advplay.model_ops.model_loaders.base_model_loader import BaseModelLoader
from advplay import paths
from sklearn.linear_model import LogisticRegression
import numpy as np

def test_load_model_success(tmp_path):
    assert available_frameworks.SKLEARN in BaseModelLoader.registry, "SklearnLoader should be registered"

    model = LogisticRegression()
    X = np.random.rand(5, 3)
    y = np.array([0, 1, 0, 1, 0])
    model.fit(X, y)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    loader_cls = BaseModelLoader.registry.get(available_frameworks.SKLEARN)
    loader = loader_cls(str(model_path))
    loaded_model = loader.load()

    assert isinstance(loaded_model, LogisticRegression)
    assert np.array_equal(model.predict(X), loaded_model.predict(X))

def test_load_model_missing_file(tmp_path):
    missing_path = tmp_path / "non_existent.joblib"
    with pytest.raises(FileNotFoundError):
        loader_cls = BaseModelLoader.registry.get(available_frameworks.SKLEARN)
        loader = loader_cls(str(missing_path))
        loaded_model = loader.load()

