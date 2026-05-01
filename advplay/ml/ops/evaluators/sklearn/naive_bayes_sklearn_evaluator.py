import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay.variables import available_frameworks, available_models

class NaiveBayesSklearnEvaluator(BaseEvaluator,
                                 framework=available_frameworks.SKLEARN,
                                 model=available_models.NAIVE_BAYES):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, X) -> pd.Series:
        X = np.asarray(X).ravel()
        X_vec = self.model.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def accuracy(self, X, y_test):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y_test)
