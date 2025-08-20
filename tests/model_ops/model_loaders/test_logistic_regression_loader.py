import pytest
import joblib
import tempfile
import os
from advplay.model_ops import registry
from advplay.model_ops.model_loaders.sklearn_model_loader import SklearnModelLoader
from advplay.variables import available_frameworks
from advplay import paths
from sklearn.linear_model import LogisticRegression
import numpy as np

def test_load_model_success(tmp_path):
    # Arrange: make sure framework is registered
    assert available_frameworks.SKLEARN in registry.BaseModelLoader.registry, "SklearnLoader should be registered"

    model = LogisticRegression()
    X = np.random.rand(5, 3)
    y = np.array([0, 1, 0, 1, 0])
    model.fit(X, y)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    # Act
    loaded_model = registry.load_model(framework=available_frameworks.SKLEARN, model_path=str(model_path))

    # Assert
    assert isinstance(loaded_model, LogisticRegression)
    assert np.array_equal(model.predict(X), loaded_model.predict(X))

def test_load_model_missing_file(tmp_path):
    missing_path = tmp_path / "non_existent.joblib"
    with pytest.raises(FileNotFoundError):
        registry.load_model(framework=available_frameworks.SKLEARN, model_path=str(missing_path))

def test_load_model_unsupported_framework(tmp_path):
    dummy_model_path = tmp_path / "dummy.joblib"
    joblib.dump({"dummy": True}, dummy_model_path)

    with pytest.raises(ValueError):
        registry.load_model(framework="nonexistent_framework", model_path=str(dummy_model_path))
