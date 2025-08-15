import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from advplay.model_ops import registry
from advplay.model_ops.evaluators.sklearn_evaluator import SklearnEvaluator
from advplay.variables import available_frameworks

DATASETS = pytest.importorskip("advplay.paths").DATASETS
SYNTHETIC_CSV = DATASETS / "synthetic_data.csv"

# ------------------- Fixtures -------------------

@pytest.fixture
def dataset():
    df = pd.read_csv(SYNTHETIC_CSV)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

@pytest.fixture
def trained_model(dataset):
    X, y = dataset
    model = LogisticRegression(solver="liblinear")
    model.fit(X, y)
    return model

@pytest.fixture
def evaluator(trained_model):
    return SklearnEvaluator(trained_model)

# ------------------- Evaluator Unit Tests -------------------

def test_predict_returns_correct_shape(evaluator, dataset):
    X, _ = dataset
    y_pred = evaluator.predict(X)
    assert isinstance(y_pred, np.ndarray), "Predict should return a numpy array"
    assert y_pred.shape[0] == X.shape[0], "Prediction length should match number of rows in X"

def test_accuracy_returns_float(evaluator, dataset):
    X, y = dataset
    acc = evaluator.accuracy(X, y)
    assert isinstance(acc, float), "Accuracy should be a float"
    assert 0.0 <= acc <= 1.0, "Accuracy should be between 0 and 1"

# ------------------- Registry / Integration Tests -------------------

def test_registry_contains_sklearn():
    evaluator_cls = registry.BaseEvaluator.registry.get(available_frameworks.SKLEARN)
    assert evaluator_cls is not None, "SklearnEvaluator should be registered"

def test_evaluate_model_accuracy_function(dataset, trained_model):
    X, y = dataset
    from advplay.model_ops import registry as reg
    acc = reg.evaluate_model_accuracy(available_frameworks.SKLEARN, trained_model, X, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
