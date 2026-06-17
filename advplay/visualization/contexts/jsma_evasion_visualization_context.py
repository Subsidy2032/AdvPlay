from dataclasses import dataclass, field
from typing import Any

import numpy as np

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext


@dataclass
class JSMAEvasionVisualizationContext(BaseVisualizationContext):
    targeted: bool = False

    # A single illustrative example (a successful flip when one exists).
    example_clean: np.ndarray = None
    example_adversarial: np.ndarray = None

    example_true_label: Any = None
    example_target_label: Any = None
    example_original_prediction: Any = None
    example_adversarial_prediction: Any = None
    # Whether the example example counts as a successful attack (drives the green/red colouring).
    example_success: bool = False

    # Distortions of the example perturbation. L0 is the number of modified pixels;
    # L2 and L∞ are measured over the raw perturbation in the model's input space.
    example_l0_norm: int = 0
    example_l2_norm: float = 0.0
    example_linf_norm: float = 0.0

    # Sparsity bookkeeping for the example, used by the "modified pixels" panel.
    example_modified_pixels: int = 0
    example_total_pixels: int = 0
    example_modified_fraction: float = 0.0

    # Modified-pixel count (L0) for every example, drives the distribution plot.
    l0_per_sample: np.ndarray = field(default_factory=lambda: np.empty(0))

    def denormalize(self, denormalizers):
        # The stored images live in the model's (normalized) input space; convert
        # them back to pixel space so they display the way a human would see them.
        self.example_clean = self._denormalize_array(denormalizers, self.example_clean[np.newaxis, ...])[0]
        self.example_adversarial = self._denormalize_array(denormalizers, self.example_adversarial[np.newaxis, ...])[0]
