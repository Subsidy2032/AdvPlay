from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.visualization.contexts.elasticnet_evasion_visualization_context import ElasticNetEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


DIFFERENCE_SCALE = 10

# (label, attribute on the context, colour) for the four distortion norms.
NORM_SPECS = (
    ("L1 distortion", "l1_per_sample", "#1f77b4"),
    ("L2 distortion", "l2_per_sample", "#2ca02c"),
    ("L∞ distortion", "linf_per_sample", "#d62728"),
    ("Elastic distortion", "elastic_per_sample", "#9467bd"),
)


class ElasticNetEvasionVisualizer(BaseVisualizer,
                                  attack_type=available_attacks.EVASION,
                                  attack_subtype=evasion_techniques.ELASTICNET):
    def visualize(self, context: ElasticNetEvasionVisualizationContext):
        out_dir = paths.VISUALIZATIONS_RESULTS / available_attacks.EVASION / \
            f"elasticnet_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._plot_example(context, out_dir)
        self._plot_distributions(context, out_dir)
        self._plot_relationships(context, out_dir)
        self._plot_heatmaps_and_sparsity(context, out_dir)

        print(f"ElasticNet visualization results saved under {out_dir}")

    def _plot_example(self, ctx, out_dir):
        clean_image, _ = _as_displayable_image(ctx.example_clean)
        adversarial_image, _ = _as_displayable_image(ctx.example_adversarial)
        difference = adversarial_image - clean_image
        # Amplify so the otherwise-imperceptible noise becomes visible.
        amplified = difference * DIFFERENCE_SCALE

        fig, (ax_clean, ax_pert, ax_adv) = plt.subplots(1, 3, figsize=(11, 4.6))

        cmap = 'gray' if clean_image.ndim == 2 else None
        ax_clean.imshow(clean_image, cmap=cmap)
        ax_clean.set_title(f"Original\nprediction = {ctx.example_original_prediction}",
                           fontsize=12, fontweight='bold', color='green')

        if amplified.ndim == 2:
            bound = float(np.max(np.abs(amplified))) if amplified.size else 0.0
            ax_pert.imshow(amplified, cmap='seismic',
                           vmin=-bound if bound else -1, vmax=bound if bound else 1)
        else:
            ax_pert.imshow(_center_for_display(amplified))
        ax_pert.set_title(f"Perturbation ×{DIFFERENCE_SCALE}", fontsize=12, fontweight='bold')

        ax_adv.imshow(adversarial_image, cmap=cmap)
        ax_adv.set_title(f"Adversarial\nprediction = {ctx.example_adversarial_prediction}",
                         fontsize=12, fontweight='bold', color='red')

        for ax in (ax_clean, ax_pert, ax_adv):
            ax.set_xticks([])
            ax.set_yticks([])

        mode = "targeted" if ctx.targeted else "untargeted"
        title = f"ElasticNet evasion ({mode})"
        if ctx.targeted:
            title += f" — target class {ctx.example_target_label}"
        title += (f"\nL1 = {ctx.example_l1_norm:.3f}   L2 = {ctx.example_l2_norm:.3f}   "
                  f"L∞ = {ctx.example_linf_norm:.3f}   elastic = {ctx.example_elastic_norm:.3f}")
        fig.suptitle(title, fontsize=13, fontweight='bold')
        fig.savefig(out_dir / "01_adversarial_example.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    def _plot_distributions(self, ctx, out_dir):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (label, attribute, colour) in zip(axes.ravel(), NORM_SPECS):
            data = np.asarray(getattr(ctx, attribute), dtype=float)
            ax.hist(data, bins=_histogram_bins(data), color=colour, alpha=0.8, edgecolor='white')
            if data.size:
                mean = float(np.mean(data))
                ax.axvline(mean, color='black', linestyle='--', linewidth=1.5,
                           label=f"mean = {mean:.3f}")
                ax.legend(fontsize=9)
            ax.set_title(f"{label} distribution", fontsize=12, fontweight='bold')
            ax.set_xlabel(label)
            ax.set_ylabel("Number of examples")

        fig.suptitle("ElasticNet distortion distributions", fontsize=14, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / "02_distortion_distributions.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    def _plot_relationships(self, ctx, out_dir):
        l1 = np.asarray(ctx.l1_per_sample, dtype=float)
        l2 = np.asarray(ctx.l2_per_sample, dtype=float)
        sparsity = np.asarray(ctx.sparsity_per_sample, dtype=float)

        fig, (ax_l1l2, ax_tradeoff) = plt.subplots(1, 2, figsize=(12, 5.2))

        # L1 vs L2 distortion relationship.
        ax_l1l2.scatter(l1, l2, s=28, alpha=0.65, color='#1f77b4', edgecolor='white', linewidth=0.5)
        if l1.size:
            mean_l1, mean_l2 = float(np.mean(l1)), float(np.mean(l2))
            ax_l1l2.axvline(mean_l1, color='gray', linestyle=':', linewidth=1)
            ax_l1l2.axhline(mean_l2, color='gray', linestyle=':', linewidth=1)
            ax_l1l2.scatter([mean_l1], [mean_l2], marker='*', s=320, color='red', zorder=5,
                            edgecolor='black', linewidth=0.6,
                            label=f"mean (L1={mean_l1:.3f}, L2={mean_l2:.3f})")
            ax_l1l2.legend(fontsize=9)
        ax_l1l2.set_title("L1 vs L2 distortion", fontsize=12, fontweight='bold')
        ax_l1l2.set_xlabel("L1 distortion")
        ax_l1l2.set_ylabel("L2 distortion")

        # Sparsity vs distortion tradeoff.
        ax_tradeoff.scatter(sparsity, l2, s=28, alpha=0.65, color='#9467bd',
                            edgecolor='white', linewidth=0.5)
        if sparsity.size:
            mean_sparsity, mean_l2 = float(np.mean(sparsity)), float(np.mean(l2))
            ax_tradeoff.axvline(mean_sparsity, color='gray', linestyle=':', linewidth=1)
            ax_tradeoff.axhline(mean_l2, color='gray', linestyle=':', linewidth=1)
            ax_tradeoff.scatter([mean_sparsity], [mean_l2], marker='*', s=320, color='red', zorder=5,
                                edgecolor='black', linewidth=0.6,
                                label=f"mean (sparsity={mean_sparsity:.1f}%, L2={mean_l2:.3f})")
            ax_tradeoff.legend(fontsize=9)
        ax_tradeoff.set_title("Sparsity vs distortion tradeoff", fontsize=12, fontweight='bold')
        ax_tradeoff.set_xlabel("Unchanged pixels (%)")
        ax_tradeoff.set_ylabel("L2 distortion")

        fig.suptitle("ElasticNet distortion relationships", fontsize=14, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / "03_distortion_relationships.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    def _plot_heatmaps_and_sparsity(self, ctx, out_dir):
        clean = np.asarray(ctx.heatmap_clean)
        adversarial = np.asarray(ctx.heatmap_adversarial)
        sparsity = np.asarray(ctx.sparsity_per_sample, dtype=float)
        num_heatmaps = len(clean)
        ncols = max(num_heatmaps, 1)

        fig = plt.figure(figsize=(max(12, 1.4 * ncols), 7.5))
        grid = fig.add_gridspec(2, ncols, height_ratios=[1, 0.85], hspace=0.4, wspace=0.15)

        last_image = None
        for i in range(num_heatmaps):
            clean_image, _ = _as_displayable_image(clean[i])
            adversarial_image, _ = _as_displayable_image(adversarial[i])
            difference = adversarial_image - clean_image
            # Per-pixel perturbation magnitude (collapse colour channels into one map).
            magnitude = np.abs(difference) if difference.ndim == 2 \
                else np.sqrt(np.sum(difference ** 2, axis=-1))

            ax = fig.add_subplot(grid[0, i])
            last_image = ax.imshow(magnitude, cmap='inferno')
            ax.set_title(f"#{i}", fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        if last_image is not None:
            fig.colorbar(last_image, ax=fig.axes[:num_heatmaps], fraction=0.025, pad=0.02)

        ax_bar = fig.add_subplot(grid[1, :])
        indices = np.arange(len(sparsity))
        ax_bar.bar(indices, sparsity, color='#2ca02c', alpha=0.85)
        if sparsity.size:
            mean_sparsity = float(np.mean(sparsity))
            ax_bar.axhline(mean_sparsity, color='black', linestyle='--', linewidth=1.5,
                           label=f"mean = {mean_sparsity:.1f}%")
            ax_bar.legend(fontsize=9)
        ax_bar.set_title("Sparsity per example (percentage of image left unchanged)",
                         fontsize=12, fontweight='bold')
        ax_bar.set_xlabel("Example index")
        ax_bar.set_ylabel("Unchanged pixels (%)")
        ax_bar.set_ylim(0, 100)

        fig.suptitle(f"ElasticNet perturbation heatmaps (first {num_heatmaps}) and sparsity",
                     fontsize=14, fontweight='bold')
        fig.savefig(out_dir / "04_perturbation_heatmaps.png", dpi=140, bbox_inches='tight')
        plt.close(fig)


def _histogram_bins(data):
    # Keep the bin count reasonable for both tiny and large sample sets.
    return int(np.clip(np.sqrt(data.size), 5, 30)) if data.size else 1


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
