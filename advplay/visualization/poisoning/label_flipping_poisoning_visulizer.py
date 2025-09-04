import numpy as np
import os
import matplotlib.pyplot as plt

from advplay.visualization.poisoning.poisoning_visualizer import PoisoningVisualizer
from advplay.variables import available_attacks, poisoning_techniques

class LabelFlippingPoisoningVisualizer(PoisoningVisualizer, attack_type=available_attacks.POISONING,
                                       attack_subtype=poisoning_techniques.LABEL_FLIPPING):
    def visualize(self):
        self.save_accuracy_graph()
        self.save_confusion_matrices()

        print(f"Results are saved to {self.directory_name}")
