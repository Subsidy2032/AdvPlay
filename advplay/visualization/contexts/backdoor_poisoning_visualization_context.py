from dataclasses import dataclass, field
import numpy as np

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext

@dataclass
class BackdoorPoisoningVisualizationContext(BaseVisualizationContext):
    base_clean_confusion_matrix: any
    base_triggered_confusion_matrix: any
    base_asr: float
    base_triggered_non_source_accuracy: float

    source_class: any
    target_class: any
    source_label: any
    target_label: any
    labels: any

    trigger: str

    portions_poisoned: list
    percentages_poisoned: list
    n_samples_poisoned: list

    clean_accuracies: list
    triggered_non_source_accuracies: list
    asrs: list

    clean_confusion_matrices: list
    triggered_confusion_matrices: list

    per_class_asr_by_portion: list
    non_target_class_labels: list

    example_clean: np.ndarray
    example_triggered: np.ndarray
    example_true_label: any
    example_clean_prediction_base: any
    example_triggered_prediction_base: any
    example_clean_prediction_poisoned: any
    example_triggered_prediction_poisoned: any
