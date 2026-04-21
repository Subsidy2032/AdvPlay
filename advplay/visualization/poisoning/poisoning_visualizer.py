from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.visualization.contexts.poisoning_visualization_context import PoisoningVisualizationContext
from advplay.variables import available_attacks
from advplay import paths

class PoisoningVisualizer(BaseVisualizer, attack_type=available_attacks.POISONING, attack_subtype=None):
    def visualize(self, context: PoisoningVisualizationContext):
        for confusion_matrix, percentage_poisoned in zip(context.confusion_matrices, context.percentages_poisoned):
            confusion_matrix = np.array(confusion_matrix)
            cf_flattened = ["{0:0.0f}\n".format(value) for value in confusion_matrix.flatten()]
            percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]

            cell_values = [f"{v1}{v2}".strip() for v1, v2 in zip(cf_flattened, percentages)]
            cell_values = np.asarray(cell_values).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])

            accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))

            figsize = plt.rcParams.get('figure.figsize')

            plt.figure(figsize=figsize)

            dark_red = LinearSegmentedColormap.from_list("DarkRed", ["#330000", "#990000", "#ff0000"])
            sns.heatmap(confusion_matrix, annot=cell_values, fmt="", cmap=dark_red, cbar=True, xticklabels=context.labels,
                        yticklabels=context.labels)

            accuracy_text = f"\n\nAccuracy={accuracy:.2f}"
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + accuracy_text)

            plt.title(f'{percentage_poisoned:.1f}% Poisoned')

            directory = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            directory_name = paths.VISUALIZATIONS_RESULTS / available_attacks.POISONING / directory
            directory_name.mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(directory_name, f"confusion_matrix_{percentage_poisoned:.1f}.png"))
            plt.close()

            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(context.percentages_poisoned, context.accuracies, marker='o', color='darkred', linewidth=2, markersize=8,
                linestyle='-')
        ax.set_xlabel("Percentage of samples poisoned (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
        ax.set_title("Model Accuracy vs Poisoned Portion", fontsize=14, fontweight='bold')
        ax.set_xticks(context.percentages_poisoned)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

        fig.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

        directory = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory_name = paths.VISUALIZATIONS_RESULTS / available_attacks.POISONING / directory
        directory_name.mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(directory_name, "poisoned_vs_accuracy.png"))
        plt.close()

        print(f"Visualization results are saved to the {directory_name} directory\n")
