from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.visualization.contexts.deepfool_evasion_visualization_context import DeepfoolEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


DIFFERENCE_SCALE = 10


class DeepfoolEvasionVisualizer(BaseVisualizer,
                                attack_type=available_attacks.EVASION,
                                attack_subtype=evasion_techniques.DEEPFOOL):
    def visualize(self, context: DeepfoolEvasionVisualizationContext):
        out_dir = paths.VISUALIZATIONS_RESULTS / available_attacks.EVASION / \
            f"deepfool_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._plot_example(context, out_dir)

        print(f"DeepFool visualization results saved under {out_dir}")

    def _plot_example(self, ctx, out_dir):
        clean_image, _ = _as_displayable_image(ctx.example_clean)
        adversarial_image, _ = _as_displayable_image(ctx.example_adversarial)
        difference = adversarial_image - clean_image
        # Amplify so the otherwise-imperceptible noise becomes visible.
        amplified = difference * DIFFERENCE_SCALE
        # Per-pixel perturbation magnitude (collapse colour channels into one map).
        magnitude = np.abs(difference) if difference.ndim == 2 else np.sqrt(np.sum(difference ** 2, axis=-1))

        fig = plt.figure(figsize=(13, 6))
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 0.28], hspace=0.25, wspace=0.15)
        ax_clean = fig.add_subplot(grid[0, 0])
        ax_adversarial = fig.add_subplot(grid[0, 1])
        ax_difference = fig.add_subplot(grid[0, 2])
        ax_heatmap = fig.add_subplot(grid[0, 3])
        ax_text = fig.add_subplot(grid[1, :])

        cmap = 'gray' if clean_image.ndim == 2 else None
        ax_clean.imshow(clean_image, cmap=cmap)
        ax_clean.set_title(f"Original\nlabel = {ctx.example_original_prediction}"
                           f"  (true = {ctx.example_true_label})",
                           fontsize=11, fontweight='bold')

        ax_adversarial.imshow(adversarial_image, cmap=cmap)
        ax_adversarial.set_title(f"Adversarial\nlabel = {ctx.example_adversarial_prediction}",
                                 fontsize=11, fontweight='bold')

        if amplified.ndim == 2:
            bound = float(np.max(np.abs(amplified))) if amplified.size else 0.0
            ax_difference.imshow(amplified, cmap='seismic',
                                 vmin=-bound if bound else -1, vmax=bound if bound else 1)
        else:
            ax_difference.imshow(_center_for_display(amplified))
        ax_difference.set_title(f"Perturbation ×{DIFFERENCE_SCALE}", fontsize=11, fontweight='bold')

        heatmap = ax_heatmap.imshow(magnitude, cmap='inferno')
        ax_heatmap.set_title(f"Magnitude heatmap\nL2 = {ctx.example_l2_norm:.4f}",
                             fontsize=11, fontweight='bold')
        fig.colorbar(heatmap, ax=ax_heatmap, fraction=0.046, pad=0.04)

        for ax in (ax_clean, ax_adversarial, ax_difference, ax_heatmap):
            ax.set_xticks([])
            ax.set_yticks([])

        ax_text.axis('off')
        summary = (
            f"Relative perturbation: {ctx.example_relative_perturbation * 100:.2f}%"
            f"        Confidence: {ctx.example_clean_confidence * 100:.2f}% "
            f"→ {ctx.example_adversarial_confidence * 100:.2f}%"
        )
        ax_text.text(0.5, 0.6, summary, ha='center', va='center',
                     fontsize=12, fontweight='bold')

        fig.suptitle("DeepFool evasion — successful flip", fontsize=14, fontweight='bold')
        fig.savefig(out_dir / "01_adversarial_example.png", dpi=140, bbox_inches='tight')
        plt.close(fig)


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
