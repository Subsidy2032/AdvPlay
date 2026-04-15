import pytest
from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.variables import available_attacks, poisoning_techniques
from advplay.visualization.contexts.poisoning_visualization_context import PoisoningVisualizationContext

@pytest.fixture
def context():
    base_acc = 0.95
    base_confusion_matrix = [[3, 7, 0], [0, 9, 0], [0, 0, 1]]

    poisoning_results = [
        {"portion": 0.1, "n_samples_poisoned": 5, "accuracy": 0.92,
         "confusion_matrix": [[3, 7, 0], [0, 9, 0], [0, 0, 11]]},
         {"portion": 0.2, "n_samples_poisoned": 10, "accuracy": 0.88,
          "confusion_matrix": [[3, 7, 0], [0, 9, 0], [0, 0, 11]]
          }]
    
    portions_poisoned = [0.0] + [poisoning_result['portion'] for poisoning_result in poisoning_results]
    percentages_poisoned = [portion * 100 for portion in portions_poisoned]
    n_samples_poisoned = [0] + [poisoning_result['n_samples_poisoned'] for poisoning_result in poisoning_results]
    accuracies = [base_acc] + [poisoning_result['accuracy'] for poisoning_result in poisoning_results]
    confusion_matrices = [base_confusion_matrix] + [poisoning_result['confusion_matrix'] for poisoning_result in poisoning_results]

    return PoisoningVisualizationContext(
        base_acc,
        base_confusion_matrix,
        1,
        2,
        [0, 1, 2],
        poisoning_results,
        portions_poisoned,
        percentages_poisoned,
        n_samples_poisoned,
        accuracies,
        confusion_matrices
    )

@pytest.fixture
def attack_info():
    return {
        "attack_type": available_attacks.POISONING,
        "technique": poisoning_techniques.LABEL_FLIPPING,
        "directory": "test_visualizations"
    }

def test_label_flipping_visualizer(context, attack_info):
    visualizer_cls = BaseVisualizer.registry.get(attack_info["attack_type"])
    visualizer = visualizer_cls()
    visualizer.visualize(context)
