from abc import ABC, abstractmethod
from datetime import datetime

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.variables import available_attacks
from advplay import paths

class PoisoningVisualizer(BaseVisualizer, ABC, attack_type=available_attacks.POISONING, attack_subtype=None):
    def __init__(self, log_file: dict, **kwargs):
        super().__init__(log_file, **kwargs)

        self.base_accuracy = log_file.get('base_accuracy')
        self.base_confusion_matrix = log_file.get('base_confusion_matrix')
        self.source_class = log_file.get('source_class')
        self.target_class = log_file.get('target_class')

        self.poisoning_results = log_file.get('poisoning_results')
        self.portions_poisoned = [0.0] + [poisoning_result['portion_to_poison'] for poisoning_result in self.poisoning_results]
        self.percentages_poisoned = [portion * 100 for portion in self.portions_poisoned]
        self.n_samples_poisoned = [0] + [poisoning_result['n_samples_poisoned'] for poisoning_result in self.poisoning_results]
        self.accuracies = [self.base_accuracy] + [poisoning_result['accuracy'] for poisoning_result in self.poisoning_results]
        self.confusion_matrices = [self.base_confusion_matrix] + [poisoning_result['confusion_matrix'] for poisoning_result in self.poisoning_results]

        self.directory_name = (paths.VISUALIZATIONS_RESULTS / available_attacks.POISONING /
                               kwargs.get('directory_name', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        self.directory_name.mkdir(parents=True, exist_ok=True)
