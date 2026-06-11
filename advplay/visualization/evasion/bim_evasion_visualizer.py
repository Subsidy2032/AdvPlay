from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.visualization.contexts.bim_evasion_visualization_context import BIMEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


DIFFERENCE_SCALE = 10


class BIMEvasionVisualizer(BaseVisualizer,
                           attack_type=available_attacks.EVASION,
                           attack_subtype=evasion_techniques.BIM):
    def visualize(self, context: BIMEvasionVisualizationContext):
        out_dir = paths.VISUALIZATIONS_RESULTS / available_attacks.EVASION / \
            f"bim_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._plot_example(context, out_dir)

        print(f"BIM visualization results saved under {out_dir}")

    def _plot_example(self, ctx, out_dir):
        clean_image, _ = _as_displayable_image(ctx.example_clean)
        adversarial_image, _ = _as_displayable_image(ctx.example_adversarial)
        # Perturbation is amplified so the otherwise-imperceptible noise becomes visible.
        difference = (adversarial_image - clean_image) * DIFFERENCE_SCALE

        fig = plt.figure(figsize=(10, 8))
        grid = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.3, wspace=0.1)
        ax_clean = fig.add_subplot(grid[0, 0])
        ax_adversarial = fig.add_subplot(grid[0, 1])
        ax_difference = fig.add_subplot(grid[0, 2])
        ax_probabilities = fig.add_subplot(grid[1, :])

        cmap = 'gray' if clean_image.ndim == 2 else None
        ax_clean.imshow(clean_image, cmap=cmap)
        ax_clean.set_title(f"Original\npredicted = {ctx.example_clean_prediction}"
                           f"  (true = {ctx.example_true_label})",
                           fontsize=11, fontweight='bold')

        ax_adversarial.imshow(adversarial_image, cmap=cmap)
        ax_adversarial.set_title(f"Adversarial\npredicted = {ctx.example_adversarial_prediction}",
                                 fontsize=11, fontweight='bold')

        diff_cmap = 'seismic' if difference.ndim == 2 else None
        diff_image = _center_for_display(difference)
        bound = float(np.max(np.abs(difference))) if difference.size else 0.0
        if difference.ndim == 2:
            ax_difference.imshow(difference, cmap=diff_cmap,
                                 vmin=-bound if bound else -1, vmax=bound if bound else 1)
        else:
            ax_difference.imshow(diff_image)
        ax_difference.set_title(f"Difference ×{DIFFERENCE_SCALE}", fontsize=11, fontweight='bold')

        for ax in (ax_clean, ax_adversarial, ax_difference):
            ax.set_xticks([])
            ax.set_yticks([])

        self._plot_probabilities(ax_probabilities, ctx)

        mode = "targeted" if ctx.targeted else "untargeted"
        title = f"BIM evasion ({mode})"
        if ctx.targeted:
            title += f" — target class {ctx.example_target_label}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.savefig(out_dir / "01_adversarial_example.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _plot_probabilities(ax, ctx):
        clean_probabilities = np.asarray(ctx.example_clean_probabilities, dtype=float)
        adversarial_probabilities = np.asarray(ctx.example_adversarial_probabilities, dtype=float)
        classes = np.arange(len(clean_probabilities))
        bar_width = 0.4

        ax.bar(classes - bar_width / 2, clean_probabilities, width=bar_width,
               color='#1f77b4', edgecolor='black', linewidth=0.5, label='Clean')
        ax.bar(classes + bar_width / 2, adversarial_probabilities, width=bar_width,
               color='#d62728', edgecolor='black', linewidth=0.5, label='Adversarial')

        # Highlight the meaningful classes.
        ax.axvline(ctx.example_true_label, color='#2ca02c', linestyle='--',
                   linewidth=1.2, alpha=0.8, label='True class')
        if ctx.targeted and ctx.example_target_label is not None:
            ax.axvline(ctx.example_target_label, color='#9467bd', linestyle=':',
                       linewidth=1.4, alpha=0.9, label='Target class')

        ax.set_xticks(classes)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax.set_title('Class probabilities: clean vs adversarial',
                     fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.legend(loc='upper right', frameon=False, fontsize=9)


def _as_displayable_image(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        side = int(round(np.sqrt(arr.size)))
        if side * side == arr.size:
            arr = arr.reshape(side, side)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr, arr.ndim in (2, 3)


def _center_for_display(difference):
    # Map a signed colour difference into a viewable [0, 1] range centred on mid-grey.
    bound = float(np.max(np.abs(difference))) if difference.size else 0.0
    if bound == 0:
        return np.full_like(difference, 0.5)
    return np.clip(0.5 + difference / (2 * bound), 0.0, 1.0)
