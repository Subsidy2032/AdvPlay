from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.visualization.contexts.jsma_evasion_visualization_context import JSMAEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


# Pixels whose perturbation magnitude is above this value are drawn as "modified".
MODIFIED_TOLERANCE = 1e-8


class JSMAEvasionVisualizer(BaseVisualizer,
                            attack_type=available_attacks.EVASION,
                            attack_subtype=evasion_techniques.JSMA):
    def visualize(self, context: JSMAEvasionVisualizationContext):
        out_dir = paths.VISUALIZATIONS_RESULTS / available_attacks.EVASION / \
            f"jsma_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._plot_example(context, out_dir)
        self._plot_l0_distribution(context, out_dir)

        print(f"JSMA visualization results saved under {out_dir}")

    def _plot_example(self, ctx, out_dir):
        clean_image, _ = _as_displayable_image(ctx.example_clean)
        adversarial_image, _ = _as_displayable_image(ctx.example_adversarial)
        difference = adversarial_image - clean_image
        # Per-pixel perturbation magnitude (collapse colour channels into one map).
        magnitude = np.abs(difference) if difference.ndim == 2 \
            else np.sqrt(np.sum(difference ** 2, axis=-1))
        # Binary "was this pixel touched" map for the rightmost panel.
        modified = (magnitude > MODIFIED_TOLERANCE).astype(float)

        fig, (ax_clean, ax_adv, ax_heat, ax_mod) = plt.subplots(1, 4, figsize=(15, 4.8))

        cmap = 'gray' if clean_image.ndim == 2 else None
        ax_clean.imshow(clean_image, cmap=cmap)
        ax_clean.set_title(f"Original\ntrue label = {ctx.example_true_label}",
                           fontsize=12, fontweight='bold')

        # Green when the attack succeeded, red when it failed.
        success_colour = 'green' if ctx.example_success else 'red'
        outcome = "SUCCESS" if ctx.example_success else "FAILURE"
        adv_title = f"Adversarial\nprediction = {ctx.example_adversarial_prediction}"
        if ctx.targeted:
            adv_title += f"\ntarget = {ctx.example_target_label}"
        adv_title += f"\n{outcome}"
        ax_adv.imshow(adversarial_image, cmap=cmap)
        ax_adv.set_title(adv_title, fontsize=12, fontweight='bold', color=success_colour)

        heat = ax_heat.imshow(magnitude, cmap='hot')
        ax_heat.set_title(f"Perturbation heatmap\nL0 = {ctx.example_l0_norm}   "
                          f"L2 = {ctx.example_l2_norm:.3f}   L∞ = {ctx.example_linf_norm:.3f}",
                          fontsize=11, fontweight='bold')
        fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04)

        ax_mod.imshow(modified, cmap='gray', vmin=0, vmax=1)
        ax_mod.set_title(f"Modified pixels (white)\n{ctx.example_modified_pixels}/"
                         f"{ctx.example_total_pixels}  "
                         f"({ctx.example_modified_fraction * 100:.2f}%)",
                         fontsize=11, fontweight='bold')

        for ax in (ax_clean, ax_adv, ax_heat, ax_mod):
            ax.set_xticks([])
            ax.set_yticks([])

        mode = "targeted" if ctx.targeted else "untargeted"
        fig.suptitle(f"JSMA evasion ({mode})", fontsize=14, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / "01_adversarial_example.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    def _plot_l0_distribution(self, ctx, out_dir):
        l0 = np.asarray(ctx.l0_per_sample, dtype=float)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(l0, bins=_histogram_bins(l0), color='#1f77b4', alpha=0.8, edgecolor='white')

        if l0.size:
            mean = float(np.mean(l0))
            median = float(np.median(l0))
            ax.axvline(mean, color='red', linestyle='--', linewidth=1.8,
                       label=f"mean = {mean:.1f}")
            ax.axvline(median, color='black', linestyle='--', linewidth=1.8,
                       label=f"median = {median:.1f}")
            ax.legend(fontsize=10)

        ax.set_title("JSMA L0 distribution (pixels modified per example)",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("L0 norm (pixels modified)")
        ax.set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(out_dir / "02_l0_distribution.png", dpi=140, bbox_inches='tight')
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
