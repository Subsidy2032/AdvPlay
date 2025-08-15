import pandas as pd
from sklearn.metrics import accuracy_score

from advplay.model_ops.evaluators.base_evaluator import BaseEvaluator
from advplay.variables import available_frameworks

class SklearnEvaluator(BaseEvaluator, framework=available_frameworks.SKLEARN):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def accuracy(self, X: pd.DataFrame, y_test: pd.Series):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y_test)
