import pandas as pd

class BaseEvaluator:
    registry = {}

    def __init_subclass__(cls, framework: str):
        if framework in BaseEvaluator.registry:
            raise ValueError(f"Subclass already registered for {framework}")

        super().__init_subclass__()
        BaseEvaluator.registry[framework] = cls

    def __init__(self, model):
        self.model = model

    def predict(self, X: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement the predict method.")

    def accuracy(self, X: pd.DataFrame, y_test: pd.Series):
        raise NotImplementedError("Subclasses must implement the accuracy method.")
