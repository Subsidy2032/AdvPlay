import pytest
import pandas as pd
import numpy as np
import tempfile
import joblib

from advplay.model_ops.trainers import base_trainer
from advplay.model_ops import registry
from advplay.variables import available_frameworks, available_training_algorithms
from advplay.model_ops.trainers.sklearn.logistic_regression_trainer import LogisticRegressionTrainer
from advplay.utils import save_model
from advplay import paths

DATASETS = paths.DATASETS
SYNTHETIC_CSV = DATASETS / "synthetic_data.csv"

# ------------------- Fixtures -------------------

@pytest.fixture
def train_test():
    dataset = pd.DataFrame({
        'feature1': np.arange(10),
        'feature2': np.arange(10, 20),
        'label': [0, 1] * 5
    })

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    return X, y

@pytest.fixture
def config():
    return {"param1": 1, "param2": True}  # Dummy config for testing

@pytest.fixture
def framework():
    return available_frameworks.SKLEARN

@pytest.fixture
def model():
    return available_training_algorithms.LOGISTIC_REGRESSION

@pytest.fixture
def trainer_cls(framework, model):
    return base_trainer.BaseTrainer.registry.get((framework, model))

@pytest.fixture
def trainer(trainer_cls, train_test, config):
    X, y = train_test
    return trainer_cls(X, y, config=config)

# ------------------- Registry Tests -------------------

def test_registry_contains_logistic_regression(trainer_cls):
    assert trainer_cls is not None, "Logistic regression should be registered under the framework"

def test_registry_unique_algorithm(framework):
    class DummyTrainer(base_trainer.BaseTrainer, framework=framework, model="logreg_dup"):
        def train(self): pass

    with pytest.raises(ValueError):
        class DuplicateTrainer(base_trainer.BaseTrainer, framework=framework, model="logreg_dup"):
            def train(self): pass

# ------------------- Initialization Tests -------------------

def test_init_type_checks(config):
    cls = base_trainer.BaseTrainer
    y = pd.Series([0, 1])
    with pytest.raises(TypeError):
        cls("not a dataframe", y, config=config)

def test_init_y_type_checks(config):
    cls = base_trainer.BaseTrainer
    X = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(TypeError):
        cls(X, "not array-like", config=config)

def test_init_alignment_error(config):
    cls = base_trainer.BaseTrainer
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 0])  # length mismatch
    with pytest.raises(ValueError):
        cls(X, y, config=config)

def test_init_empty_X(config):
    cls = base_trainer.BaseTrainer
    X = pd.DataFrame(columns=["a"])
    y = pd.Series([], dtype=int)
    with pytest.raises(ValueError):
        cls(X, y, config=config)

# ------------------- Integration Tests -------------------

def test_build_cls_returns_correct_instance(train_test, config, framework, model):
    X, y = train_test
    trainer = registry.build_trainer_cls(
        framework=framework,
        model=model,
        X_train=X,
        y_train=y,
        config=config
    )
    assert isinstance(trainer, base_trainer.BaseTrainer)

def test_train_flow(train_test, config, framework, model):
    X, y = train_test
    trainer = registry.build_trainer_cls(
        framework=framework,
        model=model,
        X_train=X,
        y_train=y,
        config=config
    )
    model = trainer.train()
    assert model is not None

# ------------------- Utils Save Model Tests -------------------

def test_utils_save_model_creates_file(train_test, tmp_path, framework):
    X, y = train_test
    trainer = LogisticRegressionTrainer(X, y, config={})
    model = trainer.train()

    original_models_path = paths.MODELS
    paths.MODELS = tmp_path

    try:
        file_path = save_model.save_model(framework, model, "test_model")
        assert file_path.exists(), "Model file should be created"

        loaded_model = joblib.load(file_path)
        assert loaded_model.__class__ == model.__class__
        assert np.array_equal(loaded_model.predict(X), model.predict(X))
    finally:
        paths.MODELS = original_models_path

def test_utils_save_model_creates_directory(train_test, tmp_path, framework):
    X, y = train_test
    trainer = LogisticRegressionTrainer(X, y, config={})
    model = trainer.train()

    original_models_path = paths.MODELS
    paths.MODELS = tmp_path / "nested_models"

    try:
        file_path = save_model.save_model( framework,model, "test_model_nested")
        assert file_path.is_file()
        assert file_path.parent.is_dir()
    finally:
        paths.MODELS = original_models_path
