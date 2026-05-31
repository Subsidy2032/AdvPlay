import torch
import numpy as np
from sklearn.metrics import accuracy_score

from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay.variables import available_frameworks

class PyTorchEvaluator(BaseEvaluator, framework=available_frameworks.PYTORCH, model=None):
    def __init__(self, model):
        super().__init__(model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        preprocessor = getattr(self.model, "preprocessor", None)
        if preprocessor is not None:
            X = preprocessor.normalize(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def accuracy(self, X: np.ndarray, y_test: np.ndarray):
        y_pred = self.predict(X)
        y_test_flat = np.ravel(y_test)
        return accuracy_score(y_test_flat, y_pred)
