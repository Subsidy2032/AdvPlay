from dataclasses import dataclass
import numpy as np

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext

@dataclass
class PoisoningVisualizationContext(BaseVisualizationContext):
    base_confusion_matrix: any
    source_class: any
    target_class: any
    labels: any

    poisoning_results: any
    portions_poisoned: any
    percentages_poisoned: any
    n_samples_poisoned: any
    accuracies: any
    confusion_matrices: any