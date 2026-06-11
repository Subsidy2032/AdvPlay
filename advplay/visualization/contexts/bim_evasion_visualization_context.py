from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext


@dataclass
class BIMEvasionVisualizationContext(BaseVisualizationContext):
    targeted: bool = False

    example_clean: np.ndarray = None
    example_adversarial: np.ndarray = None

    example_true_label: Any = None
    example_target_label: Any = None
    example_clean_prediction: Any = None
    example_adversarial_prediction: Any = None

    example_clean_probabilities: List[float] = field(default_factory=list)
    example_adversarial_probabilities: List[float] = field(default_factory=list)

    def denormalize(self, denormalizers):
        # The example images live in the model's (normalized) input space; convert
        # them back to pixel space so they display the way a human would see them.
        self.example_clean = self._denormalize_array(denormalizers, self.example_clean[np.newaxis, ...])[0]
        self.example_adversarial = self._denormalize_array(denormalizers, self.example_adversarial[np.newaxis, ...])[0]
