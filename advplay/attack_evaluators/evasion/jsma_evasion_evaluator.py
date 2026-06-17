from pathlib import Path
import numpy as np

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.evasion_evaluation_context import EvasionEvaluationContext
from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.visualization.contexts.jsma_evasion_visualization_context import JSMAEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


# Pixels whose perturbation magnitude is at or below this value are treated as unchanged.
# JSMA perturbs a small set of pixels by theta (0.1 by default), so changed pixels sit
# well above this threshold and the L0 count is not sensitive to its exact value.
MODIFIED_TOLERANCE = 1e-8


class JSMAEvasionEvaluator(BaseAttackEvaluator,
                           attack_type=available_attacks.EVASION,
                           attack_subtype=evasion_techniques.JSMA):
    def evaluate(self, context: EvasionEvaluationContext):
        model_path = context.model_path
        clean_samples = np.asarray(context.samples_data)
        adversarial_samples = np.asarray(context.perturbed_samples)
        true_labels = np.asarray(context.true_labels).ravel()

        # target_label is an array filled with the target class when the attack is
        # targeted, and None when it is untargeted.
        targeted = context.target_label is not None
        target_labels = np.asarray(context.target_label).ravel() if targeted else None

        default_path = paths.MODELS
        if not Path(model_path).is_file():
            model_path = default_path / model_path
        loader_cls = BaseModelLoader.registry.get(context.training_framework)
        loader = loader_cls(model_path, context.model, context.training_configuration)
        model = loader.load()

        evaluator_cls = BaseEvaluator.get(context.training_framework, context.model)
        evaluator = evaluator_cls(model)

        clean_predictions = np.argmax(evaluator.predict_proba(clean_samples), axis=1)
        adversarial_predictions = np.argmax(evaluator.predict_proba(adversarial_samples), axis=1)

        num_samples = len(clean_predictions)

        clean_accuracy = float(np.mean(clean_predictions == true_labels))
        adversarial_accuracy = float(np.mean(adversarial_predictions == true_labels))

        if targeted:
            # Success = the model now predicts the attacker-chosen target class.
            success_mask = adversarial_predictions == target_labels
            attack_success_rate = float(np.mean(success_mask))
        else:
            # Success = a sample the model originally classified correctly is now wrong.
            originally_correct = clean_predictions == true_labels
            success_mask = originally_correct & (adversarial_predictions != true_labels)
            if originally_correct.any():
                attack_success_rate = float(np.mean(success_mask[originally_correct]))
            else:
                attack_success_rate = 0.0

        # Perturbation distortions, measured in the model's input space (clean vs adversarial).
        perturbation = (adversarial_samples - clean_samples).reshape(num_samples, -1)
        l2_per_sample = np.linalg.norm(perturbation, ord=2, axis=1)
        linf_per_sample = np.max(np.abs(perturbation), axis=1)
        # L0 = number of modified pixels (channels collapsed: a pixel counts once even
        # if several of its colour channels move). This is JSMA's characteristic metric.
        modified_mask = _modified_pixel_mask(adversarial_samples - clean_samples)
        l0_per_sample = modified_mask.reshape(num_samples, -1).sum(axis=1)
        total_pixels = modified_mask.reshape(num_samples, -1).shape[1]

        evaluation_results = {
            "targeted": targeted,
            "target_label": int(target_labels[0]) if targeted else None,
            "num_samples": num_samples,
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "attack_success_rate": attack_success_rate,
            "average_l0_norm": float(np.mean(l0_per_sample)),
            "average_l2_norm": float(np.mean(l2_per_sample)),
            "average_linf_norm": float(np.mean(linf_per_sample)),
        }

        visualization_context = self._build_visualization_context(
            clean_samples, adversarial_samples, true_labels, target_labels, targeted,
            success_mask, clean_predictions, adversarial_predictions,
            l0_per_sample, l2_per_sample, linf_per_sample, total_pixels,
        )

        self._print_summary(evaluation_results)

        return evaluation_results, [], visualization_context

    @staticmethod
    def _build_visualization_context(clean_samples, adversarial_samples, true_labels,
                                     target_labels, targeted, success_mask,
                                     clean_predictions, adversarial_predictions,
                                     l0_per_sample, l2_per_sample, linf_per_sample,
                                     total_pixels):
        # Illustrate the attack with a successful example (fall back to the first sample).
        candidates = np.flatnonzero(success_mask)
        example_index = int(candidates[0]) if candidates.size else 0

        modified_pixels = int(l0_per_sample[example_index])
        modified_fraction = modified_pixels / total_pixels if total_pixels else 0.0

        return JSMAEvasionVisualizationContext(
            base_accuracy=None,
            targeted=targeted,
            example_clean=clean_samples[example_index],
            example_adversarial=adversarial_samples[example_index],
            example_true_label=int(true_labels[example_index]),
            example_target_label=int(target_labels[example_index]) if targeted else None,
            example_original_prediction=int(clean_predictions[example_index]),
            example_adversarial_prediction=int(adversarial_predictions[example_index]),
            example_success=bool(success_mask[example_index]),
            example_l0_norm=modified_pixels,
            example_l2_norm=float(l2_per_sample[example_index]),
            example_linf_norm=float(linf_per_sample[example_index]),
            example_modified_pixels=modified_pixels,
            example_total_pixels=total_pixels,
            example_modified_fraction=modified_fraction,
            l0_per_sample=l0_per_sample,
        )

    @staticmethod
    def _print_summary(results):
        print("JSMA evasion evaluation summary:")
        mode = "targeted" if results["targeted"] else "untargeted"
        print(f"  Mode: {mode}" + (f" (target = {results['target_label']})" if results["targeted"] else ""))
        print(f"  Samples evaluated:            {results['num_samples']}")
        print(f"  Clean accuracy:               {results['clean_accuracy'] * 100:6.2f}%")
        print(f"  Adversarial accuracy:         {results['adversarial_accuracy'] * 100:6.2f}%")
        print(f"  Attack success rate:          {results['attack_success_rate'] * 100:6.2f}%")
        print(f"  Avg L0 norm (pixels changed): {results['average_l0_norm']:.2f}")
        print(f"  Avg L2 norm:                  {results['average_l2_norm']:.4f}")
        print(f"  Avg L∞ norm:                  {results['average_linf_norm']:.4f}")


def _modified_pixel_mask(perturbation):
    # Boolean "was this pixel changed" map per sample, with colour channels collapsed so a
    # pixel counts once. Mirrors the channel heuristics the visualizer uses for display.
    magnitude = np.abs(perturbation)
    channel_axis = _channel_axis(magnitude.shape)
    if channel_axis is not None:
        magnitude = magnitude.max(axis=channel_axis)
    return magnitude > MODIFIED_TOLERANCE


def _channel_axis(shape):
    # shape is (num_samples, *image). A 3-D image carries a colour-channel axis (size 1 or 3)
    # either first (C, H, W) or last (H, W, C); anything else is treated as single-channel.
    if len(shape) != 4:
        return None
    if shape[1] in (1, 3):
        return 1
    if shape[3] in (1, 3):
        return 3
    return None
