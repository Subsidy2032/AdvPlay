import torch
import numpy as np
from sklearn.metrics import accuracy_score

from advplay.model_ops.evaluators.base_evaluator import BaseEvaluator
from advplay.variables import available_frameworks

class PyTorchEvaluator(BaseEvaluator, framework=available_frameworks.PYTORCH):
    def __init__(self, model):
        super().__init__(model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_tensor = self.model.preprocess_data(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        return y_pred

    def accuracy(self, X: np.ndarray, y_test: np.ndarray):
        y_pred = self.predict(X)
        y_test_flat = np.ravel(y_test)
        return accuracy_score(y_test_flat, y_pred)
