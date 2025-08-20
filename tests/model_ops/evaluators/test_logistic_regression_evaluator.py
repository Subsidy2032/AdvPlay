import pytest
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from advplay.model_ops import registry
from advplay.model_ops.evaluators.base_evaluator import BaseEvaluator
from advplay.model_ops.evaluators.sklearn_evaluator import SklearnEvaluator
from advplay.variables import available_frameworks

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
def trained_model(train_test):
    X, y = train_test
    model = LogisticRegression()
    model.fit(X, y)
    return model

@pytest.fixture
def evaluator(trained_model):
    return SklearnEvaluator(trained_model)

@pytest.fixture
def framework():
    return available_frameworks.SKLEARN

# ------------------- Evaluator Unit Tests -------------------

def test_predict_returns_correct_shape(evaluator, train_test):
    X, _ = train_test
    y_pred = evaluator.predict(X)
    assert isinstance(y_pred, np.ndarray), "Predict should return a numpy array"
    assert y_pred.shape[0] == X.shape[0], "Prediction length should match number of rows in X"

def test_accuracy_returns_float(evaluator, train_test):
    X, y = train_test
    acc = evaluator.accuracy(X, y)
    assert isinstance(acc, float), "Accuracy should be a float"
    assert 0.0 <= acc <= 1.0, "Accuracy should be between 0 and 1"

# ------------------- Registry / Integration Tests -------------------

def test_registry_contains_sklearn(framework):
    evaluator_cls = BaseEvaluator.registry.get(framework)
    assert evaluator_cls is not None, "SklearnEvaluator should be registered"

def test_evaluate_model_accuracy_function(train_test, trained_model, framework):
    X, y = train_test
    acc = registry.evaluate_model_accuracy(framework, trained_model, X, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
