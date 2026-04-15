from dataclasses import dataclass
import numpy as np

from advplay.attack_evaluators.contexts.base_evaluation_context import BaseEvaluationContext

@dataclass
class EvasionEvaluationContext(BaseEvaluationContext):
    model_path: any
    samples_data: np.array
    perturbed_samples: np.array
    target_label: int
