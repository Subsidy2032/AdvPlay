from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.visualization.contexts.backdoor_poisoning_visualization_context import BackdoorPoisoningVisualizationContext
from advplay.variables import available_attacks, poisoning_techniques
from advplay import paths


_BACKDOOR_CMAP = LinearSegmentedColormap.from_list(
    "BackdoorRed", ["#140507", "#4a0a14", "#a11029", "#e63946", "#ffd6c0"]
)
_COOL_CMAP = LinearSegmentedColormap.from_list(
    "BackdoorCool", ["#081229", "#12406b", "#3493c4", "#a6dcef", "#f2f9fc"]
)


class BackdoorPoisoningVisualizer(BaseVisualizer,
                                  attack_type=available_attacks.POISONING,
                                  attack_subtype=poisoning_techniques.BACKDOOR):
    def visualize(self, context: BackdoorPoisoningVisualizationContext):
        out_dir = paths.VISUALIZATIONS_RESULTS / available_attacks.POISONING / \
            f"backdoor_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._plot_metric_overview(context, out_dir)
        self._plot_stealth_vs_effectiveness(context, out_dir)
        self._plot_confusion_matrix_grid(context, out_dir)
        self._plot_per_class_asr_heatmap(context, out_dir)
        self._plot_per_class_asr_best(context, out_dir)
        self._plot_class_flow(context, out_dir)
        self._plot_trigger_exhibit(context, out_dir)

        print(f"Backdoor visualization results saved under {out_dir}")

    def _plot_metric_overview(self, ctx, out_dir):
        fig, ax = plt.subplots(figsize=(10, 6))
        percentages = ctx.percentages_poisoned

        ax.plot(percentages, ctx.clean_accuracies, marker='o', color='#1f77b4',
                linewidth=2.2, markersize=8, label='Clean accuracy')
        ax.plot(percentages, ctx.triggered_non_source_accuracies, marker='s', color='#2ca02c',
                linewidth=2.2, markersize=8, label='Trigger on non-source accuracy')
        ax.plot(percentages, ctx.asrs, marker='^', color='#d62728',
                linewidth=2.2, markersize=8, label='Attack Success Rate (ASR)')

        ax.fill_between(percentages, ctx.asrs, ctx.clean_accuracies,
                        where=[a < c for a, c in zip(ctx.asrs, ctx.clean_accuracies)],
                        color='#d62728', alpha=0.08, label='Stealth gap')

        best_idx = int(np.argmax(ctx.asrs))
        ax.scatter([percentages[best_idx]], [ctx.asrs[best_idx]], s=220,
                   facecolors='none', edgecolors='#d62728', linewidths=2.2,
                   label=f'Peak ASR ({ctx.asrs[best_idx]:.2f} @ {percentages[best_idx]:.1f}%)')

        ax.set_xlabel('Training samples poisoned (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
        title = f"Backdoor effectiveness — trigger='{ctx.trigger}'"
        if ctx.source_class is not None:
            title += f", {ctx.source_class} → {ctx.target_class}"
        else:
            title += f", any-class → {ctx.target_class}"
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(percentages)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

        fig.tight_layout()
        fig.savefig(out_dir / "01_metric_overview.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    def _plot_stealth_vs_effectiveness(self, ctx, out_dir):
        fig, ax = plt.subplots(figsize=(8, 7))
        clean_drops = [ctx.clean_accuracies[0] - a for a in ctx.clean_accuracies]
        asrs = ctx.asrs
        percentages = ctx.percentages_poisoned

        sc = ax.scatter(asrs, clean_drops, c=percentages, cmap=_BACKDOOR_CMAP,
                        s=160, edgecolors='black', linewidths=0.8)
        for asr, drop, pct in zip(asrs, clean_drops, percentages):
            ax.annotate(f"{pct:.0f}%", (asr, drop), textcoords="offset points",
                        xytext=(8, 6), fontsize=9)

        ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        ax.text(0.98, 0.02, 'Stealthy + Effective',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, color='#2ca02c', fontweight='bold', alpha=0.8)
        ax.text(0.02, 0.98, 'Conspicuous + Ineffective',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=10, color='#6c757d', alpha=0.7)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Poisoned portion (%)', fontsize=11)

        ax.set_xlabel('Attack Success Rate (higher = more effective)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clean-accuracy drop (higher = more visible)', fontsize=12, fontweight='bold')
        ax.set_title('Stealth vs. effectiveness', fontsize=13, fontweight='bold')
        ax.set_xlim(-0.02, 1.05)
        ax.grid(True, linestyle=':', alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_dir / "02_stealth_vs_effectiveness.png", dpi=140)
        plt.close(fig)

    def _plot_confusion_matrix_grid(self, ctx, out_dir):
        labels = list(ctx.labels)
        n_portions = len(ctx.percentages_poisoned)
        fig, axes = plt.subplots(n_portions, 2, figsize=(12, 4.2 * n_portions), squeeze=False)

        for row, pct in enumerate(ctx.percentages_poisoned):
            clean_cm = np.array(ctx.clean_confusion_matrices[row])
            trig_cm = np.array(ctx.triggered_confusion_matrices[row])

            self._draw_cm(axes[row][0], clean_cm, labels,
                          f"Clean input — {pct:.1f}% poisoned", _COOL_CMAP)
            self._draw_cm(axes[row][1], trig_cm, labels,
                          f"Triggered input — {pct:.1f}% poisoned", _BACKDOOR_CMAP,
                          highlight_col=ctx.target_label)

        fig.suptitle("Confusion matrices: clean vs. triggered test set",
                     fontsize=14, fontweight='bold', y=1.0)
        fig.tight_layout()
        fig.savefig(out_dir / "03_confusion_matrices.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _draw_cm(ax, cm, labels, title, cmap, highlight_col=None):
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = np.where(row_sums > 0, cm / row_sums, 0.0)
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n{norm[i, j]*100:.1f}%"

        sns.heatmap(norm, annot=annot, fmt="", cmap=cmap, cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    vmin=0.0, vmax=1.0, linewidths=0.3, linecolor='#1a1a1a')
        accuracy = np.trace(cm) / max(cm.sum(), 1)
        ax.set_title(f"{title}  (acc={accuracy:.2f})", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        if highlight_col is not None and 0 <= highlight_col < cm.shape[1]:
            ax.add_patch(plt.Rectangle((highlight_col, 0), 1, cm.shape[0],
                                       fill=False, edgecolor='#ffd000', linewidth=2.5))

    def _plot_per_class_asr_heatmap(self, ctx, out_dir):
        non_target_labels = ctx.non_target_class_labels
        if not non_target_labels:
            return

        matrix = np.zeros((len(ctx.percentages_poisoned), len(non_target_labels)), dtype=float)
        for i, per_class in enumerate(ctx.per_class_asr_by_portion):
            for j, lbl in enumerate(non_target_labels):
                val = per_class.get(lbl, np.nan)
                matrix[i, j] = np.nan if val is None else val

        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(non_target_labels)),
                                        max(4, 0.7 * len(ctx.percentages_poisoned))))
        yticklabels = [f"{p:.1f}%" for p in ctx.percentages_poisoned]
        annot = np.vectorize(lambda v: "" if np.isnan(v) else f"{v*100:.1f}%")(matrix)

        sns.heatmap(matrix, annot=annot, fmt="", cmap=_BACKDOOR_CMAP,
                    vmin=0.0, vmax=1.0, xticklabels=non_target_labels,
                    yticklabels=yticklabels, ax=ax, linewidths=0.3, linecolor='#1a1a1a',
                    cbar_kws={"label": "ASR"})
        if ctx.source_class is not None and ctx.source_class in non_target_labels:
            j = non_target_labels.index(ctx.source_class)
            ax.add_patch(plt.Rectangle((j, 0), 1, matrix.shape[0],
                                       fill=False, edgecolor='#ffd000', linewidth=2.5))
        ax.set_xlabel('Originating class (trigger applied)')
        ax.set_ylabel('Poisoned portion')
        ax.set_title('Per-class ASR across poisoning portions', fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_dir / "04_per_class_asr_heatmap.png", dpi=140)
        plt.close(fig)

    def _plot_per_class_asr_best(self, ctx, out_dir):
        non_target_labels = ctx.non_target_class_labels
        if not non_target_labels:
            return
        best_idx = int(np.argmax(ctx.asrs))
        per_class = ctx.per_class_asr_by_portion[best_idx]
        values = [per_class.get(lbl, 0.0) or 0.0 for lbl in non_target_labels]

        fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(non_target_labels)), 5))
        colors = []
        for lbl in non_target_labels:
            if ctx.source_class is not None and lbl == ctx.source_class:
                colors.append('#d62728')
            else:
                colors.append('#1f77b4')
        bars = ax.bar([str(l) for l in non_target_labels], values, color=colors,
                      edgecolor='black', linewidth=0.6)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v*100:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Fraction classified as target after trigger', fontweight='bold')
        ax.set_xlabel('True class of triggered sample', fontweight='bold')
        ax.set_title(f"Per-class attack outcomes at peak portion "
                     f"({ctx.percentages_poisoned[best_idx]:.1f}% poisoned → target={ctx.target_class})",
                     fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "05_per_class_asr_best.png", dpi=140)
        plt.close(fig)

    def _plot_class_flow(self, ctx, out_dir):
        best_idx = int(np.argmax(ctx.asrs))
        cm = np.array(ctx.triggered_confusion_matrices[best_idx])
        labels = list(ctx.labels)

        fig, ax = plt.subplots(figsize=(10, max(5, 0.7 * len(labels))))
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.invert_yaxis()
        ax.axis('off')

        left_x, right_x = 0.1, 0.9
        for idx, lbl in enumerate(labels):
            ax.text(left_x - 0.02, idx, f"{lbl}", ha='right', va='center',
                    fontsize=10, fontweight='bold')
            ax.text(right_x + 0.02, idx, f"{lbl}", ha='left', va='center',
                    fontsize=10, fontweight='bold')
            ax.plot([left_x, left_x], [idx - 0.3, idx + 0.3], color='#333', linewidth=3)
            ax.plot([right_x, right_x], [idx - 0.3, idx + 0.3], color='#333', linewidth=3)

        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = np.where(row_sums > 0, cm / row_sums, 0.0)
        for i in range(len(labels)):
            for j in range(len(labels)):
                frac = norm[i, j]
                if frac <= 0.01:
                    continue
                color = '#d62728' if j == ctx.target_label and i != ctx.target_label else '#4a6fa5'
                if ctx.source_label is not None and i == ctx.source_label and j == ctx.target_label:
                    color = '#ffb703'
                alpha = 0.25 + 0.65 * frac
                lw = 1 + 7 * frac
                ax.plot([left_x, right_x], [i, j], color=color, alpha=alpha, linewidth=lw,
                        solid_capstyle='round')

        ax.text(left_x, -0.7, "True class", ha='center', fontsize=11, fontweight='bold')
        ax.text(right_x, -0.7, "Predicted (triggered)", ha='center', fontsize=11, fontweight='bold')
        title = f"Prediction flow under trigger — {ctx.percentages_poisoned[best_idx]:.1f}% poisoned"
        ax.set_title(title, fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_dir / "06_class_flow.png", dpi=140)
        plt.close(fig)

    def _plot_trigger_exhibit(self, ctx, out_dir):
        clean = np.asarray(ctx.example_clean)
        trig = np.asarray(ctx.example_triggered)

        clean_img, ok = _as_displayable_image(clean)
        trig_img, _ = _as_displayable_image(trig)

        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
        ax_clean = fig.add_subplot(gs[0, 0])
        ax_trig = fig.add_subplot(gs[0, 1])
        ax_info = fig.add_subplot(gs[0, 2])

        if ok:
            ax_clean.imshow(clean_img, cmap='gray' if clean_img.ndim == 2 else None)
            ax_trig.imshow(trig_img, cmap='gray' if trig_img.ndim == 2 else None)
        else:
            ax_clean.plot(clean.ravel(), color='#1f77b4')
            ax_trig.plot(trig.ravel(), color='#d62728')
            ax_clean.set_ylabel('feature value')

        ax_clean.set_title(f"Clean sample\ntrue label = {ctx.example_true_label}",
                           fontsize=11, fontweight='bold')
        ax_trig.set_title(f"Triggered sample\ntrigger = '{ctx.trigger}'",
                          fontsize=11, fontweight='bold')
        for ax in (ax_clean, ax_trig):
            ax.set_xticks([])
            ax.set_yticks([])

        ax_info.axis('off')
        rows = [
            ("True label", ctx.example_true_label),
            ("Clean model → clean input", ctx.example_clean_prediction_base),
            ("Clean model → triggered input", ctx.example_triggered_prediction_base),
            ("Poisoned model → clean input", ctx.example_clean_prediction_poisoned),
            ("Poisoned model → triggered input", ctx.example_triggered_prediction_poisoned),
        ]
        y = 0.9
        ax_info.text(0.0, 1.0, "Predictions", fontsize=12, fontweight='bold',
                     transform=ax_info.transAxes)
        for name, value in rows:
            color = '#d62728' if value == ctx.target_class and 'triggered' in name.lower() \
                    and 'poisoned' in name.lower() else '#222'
            ax_info.text(0.0, y, name, fontsize=10, transform=ax_info.transAxes)
            ax_info.text(1.0, y, str(value), fontsize=10, fontweight='bold',
                         ha='right', color=color, transform=ax_info.transAxes)
            y -= 0.12

        success = (ctx.example_triggered_prediction_poisoned == ctx.target_class
                   and ctx.example_clean_prediction_poisoned != ctx.target_class)
        verdict = "attack succeeded" if success else "attack did not fire on this sample"
        ax_info.text(0.0, 0.12, f"Verdict: {verdict}", fontsize=11, fontweight='bold',
                     color='#d62728' if success else '#555', transform=ax_info.transAxes)

        fig.suptitle("Trigger exhibit: how the poisoned model reacts to the backdoor",
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_dir / "07_trigger_exhibit.png", dpi=140)
        plt.close(fig)


def _as_displayable_image(x):
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim not in (2, 3):
        return arr, False
    if arr.dtype.kind == 'f':
        max_val = float(np.nanmax(arr)) if arr.size else 1.0
        if max_val > 1.5:
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        else:
            arr = np.clip(arr, 0.0, 1.0)
    return arr, True
