from dataclasses import dataclass

@dataclass
class BaseEvaluationContext:
    model: any
    X_test: any
    y_test: any
    training_framework: any
    training_configuration: any