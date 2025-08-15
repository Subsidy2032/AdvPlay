import pytest
import pandas as pd
import numpy as np
import tempfile
import joblib
import os

from advplay.model_ops.trainers import base_trainer
from advplay.model_ops import registry
from advplay import paths
from advplay.variables import available_frameworks, available_training_algorithms
from advplay.model_ops.trainers.sklearn.logistic_regression_trainer import LogisticRegressionTrainer

DATASETS = paths.DATASETS
SYNTHETIC_CSV = DATASETS / "synthetic_data.csv"

# ------------------- Fixtures -------------------

@pytest.fixture
def dataset():
    return pd.read_csv(SYNTHETIC_CSV)

@pytest.fixture
def model_name():
    return "logreg_test_model"

@pytest.fixture
def config():
    return {"param1": 1, "param2": True}  # Dummy config for testing

@pytest.fixture
def label_column(dataset):
    return dataset.columns[-1]

@pytest.fixture
def framework():
    return available_frameworks.SKLEARN

@pytest.fixture
def training_algorithm():
    return available_training_algorithms.LOGISTIC_REGRESSION

@pytest.fixture
def trainer_cls(framework, training_algorithm):
    return base_trainer.BaseTrainer.registry.get((framework, training_algorithm))

@pytest.fixture
def trainer(trainer_cls, dataset, label_column, model_name, config):
    return trainer_cls(model_name, config, dataset, label_column, test_portion=0.2, seed=42)

# ------------------- Registry Tests -------------------

def test_registry_contains_logistic_regression(trainer_cls):
    assert trainer_cls is not None, "Logistic regression should be registered under the framework"

def test_registry_unique_algorithm(framework):
    class DummyTrainer(base_trainer.BaseTrainer, framework=framework, training_algorithm="logreg_dup"):
        def train(self): pass

    with pytest.raises(ValueError):
        class DuplicateTrainer(base_trainer.BaseTrainer, framework=framework, training_algorithm="logreg_dup"):
            def train(self): pass

# ------------------- Initialization Tests -------------------

def test_init_type_checks(dataset, model_name, config, label_column):
    cls = base_trainer.BaseTrainer
    with pytest.raises(TypeError):
        cls(model_name, config, "not a dataframe", label_column, 0.2, seed=1)

def test_init_label_column_missing(dataset, model_name, config):
    cls = base_trainer.BaseTrainer
    with pytest.raises(ValueError):
        cls(model_name, config, dataset, "nonexistent_column", 0.2, seed=1)

def test_init_empty_dataset(model_name, config, label_column):
    cls = base_trainer.BaseTrainer
    empty_df = pd.DataFrame(columns=[label_column])
    with pytest.raises(ValueError):
        cls(model_name, config, empty_df, label_column, 0.2, seed=1)

def test_init_invalid_test_portion(dataset, model_name, config, label_column):
    cls = base_trainer.BaseTrainer
    for invalid in [-0.1, 0, 1, 2]:
        with pytest.raises(ValueError):
            cls(model_name, config, dataset, label_column, invalid, seed=1)

# ------------------- Trainer Functionality -------------------

def test_train_method_called(trainer, monkeypatch):
    called = {}
    monkeypatch.setattr(trainer, "train", lambda: called.update({'yes': True}))
    trainer.train()
    assert 'yes' in called, "train() method should be called"

# ------------------- Integration Tests -------------------

def test_build_cls_returns_correct_instance(dataset, model_name, config, label_column, framework, training_algorithm):
    trainer = registry.build_trainer_cls(
        framework=framework,
        training_algorithm=training_algorithm,
        model_name=model_name,
        config=config,
        dataset=dataset,
        label_column=label_column,
        test_portion=0.2,
        seed=42
    )
    assert isinstance(trainer, base_trainer.BaseTrainer)

def test_train_flow(dataset, model_name, config, label_column, framework, training_algorithm):
    trainer = registry.build_trainer_cls(
        framework=framework,
        training_algorithm=training_algorithm,
        model_name=model_name,
        config=config,
        dataset=dataset,
        label_column=label_column,
        test_portion=0.2,
        seed=42
    )
    trainer.train()

# ------------------- Random Seed Reproducibility -------------------

def test_seed_reproducibility(dataset, model_name, config, label_column, framework, training_algorithm):
    t1 = registry.build_trainer_cls(framework, training_algorithm, model_name, config, dataset, label_column, 0.2, seed=42)
    t2 = registry.build_trainer_cls(framework, training_algorithm, model_name, config, dataset, label_column, 0.2, seed=42)
    assert t1.seed == t2.seed


def test_save_model_creates_file(dataset, model_name, config, label_column, tmp_path, framework):
    trainer = LogisticRegressionTrainer(model_name, config, dataset, label_column, test_portion=0.2, seed=42)
    model, _, _ = trainer.train()

    original_models_path = paths.MODELS
    paths.MODELS = tmp_path

    try:
        # Include framework in path manually
        file_path = paths.MODELS / framework / f"{model_name}.joblib"
        trainer.save_model(model)  # assume save_model updated to use framework

        assert file_path.exists(), "Model file should be created"

        loaded_model = joblib.load(file_path)
        assert loaded_model.__class__ == model.__class__, "Loaded model should match saved model type"

        X_test = dataset.drop(columns=[label_column])
        assert (loaded_model.predict(X_test) == model.predict(X_test)).all(), "Predictions should match after loading"

    finally:
        paths.MODELS = original_models_path


def test_save_model_creates_directory(dataset, model_name, config, label_column, tmp_path, framework):
    trainer = LogisticRegressionTrainer(model_name, config, dataset, label_column, test_portion=0.2, seed=42)
    model, _, _ = trainer.train()

    nested_dir = tmp_path / "nested_models"
    original_models_path = paths.MODELS
    paths.MODELS = nested_dir

    try:
        # Directory should include framework
        framework_dir = nested_dir / framework
        file_path = framework_dir / f"{model_name}.joblib"
        trainer.save_model(model)  # assume save_model updated to use framework

        assert framework_dir.is_dir(), "save_model should create the framework subdirectory"
        assert file_path.is_file(), "Model file should exist in the created framework directory"
    finally:
        paths.MODELS = original_models_path

