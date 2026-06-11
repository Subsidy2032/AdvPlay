from dataclasses import dataclass
from typing import Any

import numpy as np

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext


@dataclass
class CWEvasionVisualizationContext(BaseVisualizationContext):
    targeted: bool = False

    # A single illustrative example (a successful targeted flip when targeted).
    example_clean: np.ndarray = None
    example_adversarial: np.ndarray = None

    example_true_label: Any = None
    example_target_label: Any = None
    example_original_prediction: Any = None
    example_adversarial_prediction: Any = None

    example_l2_norm: float = 0.0
    example_relative_perturbation: float = 0.0
    example_clean_confidence: float = 0.0
    example_adversarial_confidence: float = 0.0

    def denormalize(self, denormalizers):
        # The example images live in the model's (normalized) input space; convert
        # them back to pixel space so they display the way a human would see them.
        self.example_clean = self._denormalize_array(denormalizers, self.example_clean[np.newaxis, ...])[0]
        self.example_adversarial = self._denormalize_array(denormalizers, self.example_adversarial[np.newaxis, ...])[0]
