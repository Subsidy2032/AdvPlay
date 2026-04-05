from dataclasses import dataclass
import numpy as np

from advplay.attack_evaluators.contexts.base_evaluation_context import BaseEvaluationContext

@dataclass
class PoisoningContext(BaseEvaluationContext):
    clean_dataset: np.array
    poisoned_datasets: np.array
    model_name: str
