from dataclasses import dataclass, field
from typing import Any

import numpy as np

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext


@dataclass
class ElasticNetEvasionVisualizationContext(BaseVisualizationContext):
    targeted: bool = False

    # A single illustrative example (a successful flip when one exists).
    example_clean: np.ndarray = None
    example_adversarial: np.ndarray = None

    example_true_label: Any = None
    example_target_label: Any = None
    example_original_prediction: Any = None
    example_adversarial_prediction: Any = None

    example_l1_norm: float = 0.0
    example_l2_norm: float = 0.0
    example_linf_norm: float = 0.0
    example_elastic_norm: float = 0.0

    # Per-sample distortions (measured in the model's input space) for every example.
    # These drive the distribution, relationship and sparsity plots.
    l1_per_sample: np.ndarray = field(default_factory=lambda: np.empty(0))
    l2_per_sample: np.ndarray = field(default_factory=lambda: np.empty(0))
    linf_per_sample: np.ndarray = field(default_factory=lambda: np.empty(0))
    elastic_per_sample: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Percentage of pixels left unchanged by the attack, per example.
    sparsity_per_sample: np.ndarray = field(default_factory=lambda: np.empty(0))

    # A subset of clean/adversarial pairs (at most 10) whose perturbations are
    # rendered as heatmaps. Stored as pairs so the difference can be taken after
    # the images have been denormalized back into pixel space.
    heatmap_clean: np.ndarray = field(default_factory=lambda: np.empty(0))
    heatmap_adversarial: np.ndarray = field(default_factory=lambda: np.empty(0))

    def denormalize(self, denormalizers):
        # The stored images live in the model's (normalized) input space; convert
        # them back to pixel space so they display the way a human would see them.
        self.example_clean = self._denormalize_array(denormalizers, self.example_clean[np.newaxis, ...])[0]
        self.example_adversarial = self._denormalize_array(denormalizers, self.example_adversarial[np.newaxis, ...])[0]

        if self.heatmap_clean is not None and len(self.heatmap_clean):
            self.heatmap_clean = self._denormalize_array(denormalizers, self.heatmap_clean)
            self.heatmap_adversarial = self._denormalize_array(denormalizers, self.heatmap_adversarial)
