import pytest
from advplay.visualization.visualizer import visualizer
from advplay.variables import available_attacks, poisoning_techniques

@pytest.fixture
def valid_log_data():
    # Minimal valid log structure for your visualizer
    return {
        "base_accuracy": 0.95,
        "base_confusion_matrix": [[3, 7, 0],
                                  [0, 9, 0],
                                  [0, 0, 11]],
        "source_class": None,
        "target_class": None,
        "poisoning_results": [
            {"portion_to_poison": 0.1, "n_samples_poisoned": 5, "accuracy": 0.92,
             "confusion_matrix": [[3, 7, 0], [0, 9, 0], [0, 0, 11]]},
            {"portion_to_poison": 0.2, "n_samples_poisoned": 10, "accuracy": 0.88,
             "confusion_matrix": [[3, 7, 0], [0, 9, 0], [0, 0, 11]]}
        ]
    }

@pytest.fixture
def attack_info():
    return {
        "attack_type": available_attacks.POISONING,
        "attack_subtype": poisoning_techniques.LABEL_FLIPPING,
        "directory_name": "test_visualizations"
    }

def test_label_flipping_visualizer(valid_log_data, attack_info):
    visualizer(
        attack_type=attack_info["attack_type"],
        attack_subtype=attack_info["attack_subtype"],
        log_file=valid_log_data,
        directory_name=attack_info["directory_name"]
    )
