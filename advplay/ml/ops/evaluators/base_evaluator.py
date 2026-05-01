import pandas as pd

class BaseEvaluator:
    registry = {}

    def __init_subclass__(cls, framework: str, model: str = None):
        key = (framework, model)
        if key in BaseEvaluator.registry:
            raise ValueError(f"Subclass already registered for {framework} + {model}")

        super().__init_subclass__()
        BaseEvaluator.registry[key] = cls

    @classmethod
    def get(cls, framework: str, model: str = None):
        evaluator_cls = cls.registry.get((framework, model))
        if evaluator_cls is None:
            evaluator_cls = cls.registry.get((framework, None))
        return evaluator_cls

    def __init__(self, model):
        self.model = model

    def predict(self, X: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement the predict method.")

    def accuracy(self, X: pd.DataFrame, y_test: pd.Series):
        raise NotImplementedError("Subclasses must implement the accuracy method.")
