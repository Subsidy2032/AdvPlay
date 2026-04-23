from dataclasses import dataclass
import numpy as np

from advplay.attack_evaluators.contexts.base_evaluation_context import BaseEvaluationContext

@dataclass
class BackdoorPoisoningEvaluationContext(BaseEvaluationContext):
    clean_dataset: dict
    poisoned_datasets: dict
    model_name: str

    X_test_triggered: np.ndarray

    source_label: any
    target_label: any
    source_class: any
    target_class: any
    labels: any

    trigger: str
    example_clean: np.ndarray
    example_triggered: np.ndarray
    example_true_label: any
