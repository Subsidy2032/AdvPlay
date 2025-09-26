from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.variables import available_attacks
from advplay import paths

class PoisoningVisualizer(BaseVisualizer, ABC, attack_type=available_attacks.POISONING, attack_subtype=None):
    def __init__(self, log_file: dict, **kwargs):
        super().__init__(log_file, **kwargs)

        self.base_accuracy = log_file.get('base_accuracy')
        self.base_confusion_matrix = log_file.get('base_confusion_matrix')
        self.source_class = log_file.get('source')
        self.target_class = log_file.get('target')
        self.labels = log_file.get('labels')

        self.poisoning_results = log_file.get('poisoning_results')
        self.portions_poisoned = [0.0] + [poisoning_result['portion'] for poisoning_result in self.poisoning_results]
        self.percentages_poisoned = [portion * 100 for portion in self.portions_poisoned]
        self.n_samples_poisoned = [0] + [poisoning_result['n_samples_poisoned'] for poisoning_result in self.poisoning_results]
        self.accuracies = [self.base_accuracy] + [poisoning_result['accuracy'] for poisoning_result in self.poisoning_results]
        self.confusion_matrices = [self.base_confusion_matrix] + [poisoning_result['confusion_matrix'] for poisoning_result in self.poisoning_results]

        self.directory_name = (paths.VISUALIZATIONS_RESULTS / available_attacks.POISONING /
                               kwargs.get('directory', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        self.directory_name.mkdir(parents=True, exist_ok=True)

    def save_confusion_matrices(self):
        for confusion_matrix, percentage_poisoned in zip(self.confusion_matrices, self.percentages_poisoned):
            confusion_matrix = np.array(confusion_matrix)
            cf_flattened = ["{0:0.0f}\n".format(value) for value in confusion_matrix.flatten()]
            percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]

            cell_values = [f"{v1}{v2}".strip() for v1, v2 in zip(cf_flattened, percentages)]
            cell_values = np.asarray(cell_values).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])

            accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))

            figsize = plt.rcParams.get('figure.figsize')

            plt.figure(figsize=figsize)

            dark_red = LinearSegmentedColormap.from_list("DarkRed", ["#330000", "#990000", "#ff0000"])
            sns.heatmap(confusion_matrix, annot=cell_values, fmt="", cmap=dark_red, cbar=True, xticklabels=self.labels,
                        yticklabels=self.labels)

            accuracy_text = f"\n\nAccuracy={accuracy:.2f}"
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + accuracy_text)

            plt.title(f'{percentage_poisoned:.1f}% Poisoned')

            plt.savefig(os.path.join(self.directory_name, f"confusion_matrix_{percentage_poisoned:.1f}.png"))
            plt.close()

    def save_accuracy_graph(self):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.percentages_poisoned, self.accuracies, marker='o', color='darkred', linewidth=2, markersize=8,
                linestyle='-')
        ax.set_xlabel("Percentage of samples poisoned (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
        ax.set_title("Model Accuracy vs Poisoned Portion", fontsize=14, fontweight='bold')
        ax.set_xticks(self.percentages_poisoned)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

        fig.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        plt.savefig(os.path.join(self.directory_name, "poisoned_vs_accuracy.png"))
        plt.close()
