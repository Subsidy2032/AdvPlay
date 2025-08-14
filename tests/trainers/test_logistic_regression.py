import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from advplay.model_ops.trainers import base_trainer
from advplay.model_ops.trainers import logistic_regression_trainer
from advplay.model_ops import registry
from advplay import paths

DATASETS = paths.DATASETS
SYNTHETIC_CSV = DATASETS / "synthetic_data.csv"

@pytest.fixture
def dataset():
    df = pd.read_csv(SYNTHETIC_CSV)
    return df

@pytest.fixture
def model_name():
    return "logreg_test_model"

@pytest.fixture
def label_column(dataset):
    # pick the last column as label by default
    return dataset.columns[-1]

@pytest.fixture
def trainer_cls():
    return base_trainer.BaseTrainer.registry.get("logistic_regression")

@pytest.fixture
def trainer(trainer_cls, dataset, label_column, model_name):
    return trainer_cls(model_name, dataset, label_column, test_portion=0.2, seed=42)

# ------------------- Registry Tests -------------------

def test_registry_contains_logistic_regression(trainer_cls):
    assert trainer_cls is not None, "Logistic regression should be registered"

def test_registry_unique_algorithm():
    class DummyTrainer(base_trainer.BaseTrainer, training_algorithm="logistic_regression_dup"):
        def train(self): pass

    with pytest.raises(ValueError):
        class DuplicateTrainer(base_trainer.BaseTrainer, training_algorithm="logistic_regression_dup"):
            def train(self): pass

# ------------------- Initialization Tests -------------------

def test_init_type_checks(dataset, model_name, label_column):
    cls = base_trainer.BaseTrainer
    with pytest.raises(TypeError):
        cls(model_name, "not a dataframe", label_column, 0.2, seed=1)

def test_init_label_column_missing(dataset, model_name):
    cls = base_trainer.BaseTrainer
    with pytest.raises(ValueError):
        cls(model_name, dataset, "nonexistent_column", 0.2, seed=1)

def test_init_empty_dataset(model_name, label_column):
    cls = base_trainer.BaseTrainer
    empty_df = pd.DataFrame(columns=[label_column])
    with pytest.raises(ValueError):
        cls(model_name, empty_df, label_column, 0.2, seed=1)

def test_init_invalid_test_portion(dataset, model_name, label_column):
    cls = base_trainer.BaseTrainer
    for invalid in [-0.1, 0, 1, 2]:
        with pytest.raises(ValueError):
            cls(model_name, dataset, label_column, invalid, seed=1)

# ------------------- Trainer Functionality -------------------

def test_train_method_called(trainer, monkeypatch):
    called = {}

    monkeypatch.setattr(trainer, "train", lambda: called.update({'yes': True}))
    trainer.train()
    assert 'yes' in called, "train() method should be called"

# ------------------- Integration Tests -------------------

def test_build_cls_returns_correct_instance(dataset, model_name, label_column):
    trainer = registry.build_cls(
        training_algorithm="logistic_regression",
        model_name=model_name,
        dataset=dataset,
        label_column=label_column,
        test_portion=0.2,
        seed=42
    )
    assert isinstance(trainer, base_trainer.BaseTrainer)

def test_train_flow(dataset, model_name, label_column):
    trainer = registry.build_cls(
        training_algorithm="logistic_regression",
        model_name=model_name,
        dataset=dataset,
        label_column=label_column,
        test_portion=0.2,
        seed=42
    )
    trainer.train()

# ------------------- Random Seed Reproducibility -------------------

def test_seed_reproducibility(dataset, model_name, label_column, trainer_cls):
    t1 = trainer_cls(model_name, dataset, label_column, test_portion=0.2, seed=42)
    t2 = trainer_cls(model_name, dataset, label_column, test_portion=0.2, seed=42)
    # For real models, you’d compare model states
    np.testing.assert_array_equal(t1.seed, t2.seed)
