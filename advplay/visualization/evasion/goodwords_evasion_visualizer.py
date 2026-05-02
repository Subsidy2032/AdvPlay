from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.visualization.contexts.goodwords_evasion_visualization_context import GoodwordsEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


_DARK_RED_CMAP = LinearSegmentedColormap.from_list(
    "DarkRed", ["#330000", "#990000", "#ff0000"]
)


class GoodwordsEvasionVisualizer(BaseVisualizer,
                                 attack_type=available_attacks.EVASION,
                                 attack_subtype=evasion_techniques.GOODWORDS):
    def visualize(self, context: GoodwordsEvasionVisualizationContext):
        out_dir = paths.VISUALIZATIONS_RESULTS / available_attacks.EVASION / \
            f"goodwords_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._plot_evasion_curve(context, out_dir)
        self._plot_top_contributions(context, out_dir)
        self._plot_per_message_probabilities(context, out_dir)

        print(f"Goodwords visualization results saved under {out_dir}")

    @staticmethod
    def _plot_evasion_curve(ctx, out_dir):
        word_counts = list(ctx.word_counts)
        rates = list(ctx.evasion_rates)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(word_counts, rates, marker='o', color='darkred', linewidth=2,
                markersize=8, linestyle='-', label='Evasion rate')

        for x, y in zip(word_counts, rates):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=9, color='#333')

        ax.axhline(50, color='#6c757d', linestyle='--', linewidth=1.2,
                   alpha=0.8, label='50% threshold')
        ax.axhline(90, color='#2ca02c', linestyle='--', linewidth=1.2,
                   alpha=0.8, label='90% threshold')

        if rates:
            rounded = [round(r, 1) for r in rates]
            max_rounded = max(rounded)
            best_idx = next(i for i, r in enumerate(rounded) if r == max_rounded)
            best_x = word_counts[best_idx]
            best_y = rates[best_idx]
            ax.scatter([best_x], [best_y], s=240, facecolors='none',
                       edgecolors='darkred', linewidths=2.2, zorder=5)
            ax.annotate(
                f"max {best_y:.1f}%\n@ {best_x} words",
                xy=(best_x, best_y),
                xytext=(15, -25),
                textcoords="offset points",
                fontsize=10, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.8),
            )

        ax.set_xlabel('Number of goodwords appended', fontsize=12, fontweight='bold')
        ax.set_ylabel('Evasion rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f"Goodwords evasion: {ctx.source} → {ctx.target}",
                     fontsize=14, fontweight='bold')
        ax.set_xticks(word_counts)
        ax.set_ylim(-2, 112)
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)
        ax.legend(loc='lower right', frameon=False)

        fig.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        fig.savefig(out_dir / "01_evasion_rate_vs_words.png", dpi=140)
        plt.close(fig)

    @staticmethod
    def _plot_top_contributions(ctx, out_dir):
        contributions = list(ctx.top_word_contributions)
        if not contributions:
            return

        words = [w for w, _ in contributions]
        values = [v for _, v in contributions]

        fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(words))))
        y_positions = np.arange(len(words))
        norm_values = np.array(values, dtype=float)
        if norm_values.max() > norm_values.min():
            shades = (norm_values - norm_values.min()) / (norm_values.max() - norm_values.min())
        else:
            shades = np.full_like(norm_values, 0.6)
        colors = [_DARK_RED_CMAP(0.25 + 0.7 * s) for s in shades]

        bars = ax.barh(y_positions, values, color=colors,
                       edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:.2f}%",
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(f'Avg {ctx.source}-class probability reduction (%)',
                      fontsize=12, fontweight='bold')
        ax.set_ylabel('Goodword', fontsize=12, fontweight='bold')
        ax.set_title(f"Top contributing goodwords (target = {ctx.target})",
                     fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', linestyle=':', linewidth=0.8, alpha=0.7)

        fig.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1)
        fig.savefig(out_dir / "02_top_word_contributions.png", dpi=140)
        plt.close(fig)

    @staticmethod
    def _plot_per_message_probabilities(ctx, out_dir):
        if not ctx.example_messages or not ctx.representative_word_counts:
            return

        counts = ctx.representative_word_counts
        probs_per_message = np.array(ctx.per_message_source_probs)
        n_messages = len(probs_per_message)
        n_counts = len(counts)
        if n_messages == 0 or n_counts == 0:
            return

        fig, ax = plt.subplots(figsize=(max(8, 1.1 * n_messages), 5))

        group_centers = np.arange(n_messages)
        bar_width = 0.8 / n_counts
        count_colors = [_DARK_RED_CMAP(0.2 + 0.75 * (i / max(n_counts - 1, 1)))
                        for i in range(n_counts)]

        for j, (count, color) in enumerate(zip(counts, count_colors)):
            offsets = group_centers - 0.4 + bar_width * (j + 0.5)
            ax.bar(offsets, probs_per_message[:, j], width=bar_width,
                   color=color, edgecolor='black', linewidth=0.5,
                   label=f"+{count} words")

        ax.axhline(0.5, color='#6c757d', linestyle='--', linewidth=1.2,
                   alpha=0.8, label='0.5 threshold')

        ax.set_xticks(group_centers)
        ax.set_xticklabels([f"Msg {i + 1}" for i in range(n_messages)], fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel(f'{ctx.source}-class messages', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'P({ctx.source})', fontsize=12, fontweight='bold')
        ax.set_title(f"Source-class probability per message vs. goodwords appended",
                     fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

        fig.tight_layout()
        plt.subplots_adjust(left=0.08, right=0.82, top=0.9, bottom=0.15)
        fig.savefig(out_dir / "03_per_message_probabilities.png", dpi=140)
        plt.close(fig)
