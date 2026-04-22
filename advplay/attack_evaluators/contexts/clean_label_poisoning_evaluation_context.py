from dataclasses import dataclass
import numpy as np

from advplay.attack_evaluators.contexts.base_evaluation_context import BaseEvaluationContext

@dataclass
class CleanLabelPoisoningEvaluationContext(BaseEvaluationContext):
    model_path: any
    X_poisoned: np.ndarray
    y: np.ndarray
    indices_to_poison: np.ndarray
    target_sample: np.ndarray
    target_label: int
    model_name: str
