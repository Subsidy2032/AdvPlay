from dataclasses import dataclass, field
from typing import Any, List, Tuple
import numpy as np

from advplay.attack_evaluators.contexts.evasion_evaluation_context import EvasionEvaluationContext


@dataclass
class GoodwordsEvasionEvaluationContext(EvasionEvaluationContext):
    true_labels: np.ndarray = None
    source: Any = None
    target: Any = None
    source_class_idx: int = None
    target_class_idx: int = None

    ranked_goodwords: List[str] = field(default_factory=list)
    word_contributions: List[Tuple[str, float]] = field(default_factory=list)

    word_counts: List[int] = field(default_factory=list)
    rates_per_count: List[dict] = field(default_factory=list)
