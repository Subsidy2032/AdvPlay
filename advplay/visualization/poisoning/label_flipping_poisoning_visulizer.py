import numpy as np
import os
import matplotlib.pyplot as plt

from advplay.visualization.poisoning.poisoning_visualizer import PoisoningVisualizer
from advplay.variables import available_attacks, poisoning_techniques

class LabelFlippingPoisoningVisualizer(PoisoningVisualizer, attack_type=available_attacks.POISONING,
                                       attack_subtype=poisoning_techniques.LABEL_FLIPPING):
    def __init__(self, log_file, **kwargs):
        super().__init__(log_file, **kwargs)

    def visualize(self):
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(self.percentages_poisoned, self.accuracies, marker='o', color='darkred', linewidth=2, markersize=8, linestyle='--')
        ax.set_xlabel("Percentage of samples poisoned (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
        ax.set_title("Model Accuracy vs Poisoned Portion", fontsize=14, fontweight='bold')
        ax.set_xticks(self.percentages_poisoned)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

        # Optional: annotate each point with its value
        for x, y in zip(self.portions_poisoned, self.accuracies):
            ax.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=10)

        fig.tight_layout()
        plt.savefig(os.path.join(self.directory_name, "poisoned_vs_accuracy.png"))
        plt.close()

        print(f"Results are saved to {self.directory_name}")
