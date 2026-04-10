import pytest
from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.variables import available_attacks, poisoning_techniques

@pytest.fixture
def valid_log_data():
    # Minimal valid log structure for your visualizer
    return {
        "result": {
            "technique": poisoning_techniques.LABEL_FLIPPING,
            "labels": [0, 1, 2],
        },
        "evaluation": {
            "base_accuracy": 0.95,
            "base_confusion_matrix": [[3, 7, 0],
                                      [0, 9, 0],
                                      [0, 0, 11]],
            "poisoning_results": [
                {"portion": 0.1, "n_samples_poisoned": 5, "accuracy": 0.92,
                 "confusion_matrix": [[3, 7, 0], [0, 9, 0], [0, 0, 11]]},
                 {"portion": 0.2, "n_samples_poisoned": 10, "accuracy": 0.88,
                  "confusion_matrix": [[3, 7, 0], [0, 9, 0], [0, 0, 11]]
        }]}
        }

@pytest.fixture
def attack_info():
    return {
        "attack_type": available_attacks.POISONING,
        "technique": poisoning_techniques.LABEL_FLIPPING,
        "directory": "test_visualizations"
    }

def test_label_flipping_visualizer(valid_log_data, attack_info):
    key = (attack_info["attack_type"], attack_info["technique"])
    visualizer_cls = BaseVisualizer.registry.get(key)
    visualizer = visualizer_cls(valid_log_data, directory=attack_info["directory"])
    visualizer.visualize()
