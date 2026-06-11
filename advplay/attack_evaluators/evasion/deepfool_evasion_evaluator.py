from collections import Counter
from pathlib import Path
import numpy as np

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.evasion_evaluation_context import EvasionEvaluationContext
from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.visualization.contexts.deepfool_evasion_visualization_context import DeepfoolEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


class DeepfoolEvasionEvaluator(BaseAttackEvaluator,
                               attack_type=available_attacks.EVASION,
                               attack_subtype=evasion_techniques.DEEPFOOL):
    def evaluate(self, context: EvasionEvaluationContext):
        model_path = context.model_path
        clean_samples = np.asarray(context.samples_data)
        adversarial_samples = np.asarray(context.perturbed_samples)
        true_labels = np.asarray(context.true_labels).ravel()

        default_path = paths.MODELS
        if not Path(model_path).is_file():
            model_path = default_path / model_path
        loader_cls = BaseModelLoader.registry.get(context.training_framework)
        loader = loader_cls(model_path, context.model, context.training_configuration)
        model = loader.load()

        evaluator_cls = BaseEvaluator.get(context.training_framework, context.model)
        evaluator = evaluator_cls(model)

        clean_probabilities = evaluator.predict_proba(clean_samples)
        adversarial_probabilities = evaluator.predict_proba(adversarial_samples)
        clean_predictions = np.argmax(clean_probabilities, axis=1)
        adversarial_predictions = np.argmax(adversarial_probabilities, axis=1)

        num_samples = len(clean_predictions)
        sample_indices = np.arange(num_samples)

        # Success = a sample the model originally classified correctly is now wrong.
        originally_correct = clean_predictions == true_labels
        success_mask = originally_correct & (adversarial_predictions != true_labels)
        if originally_correct.any():
            attack_success_rate = float(np.mean(success_mask[originally_correct]))
        else:
            attack_success_rate = 0.0

        # Perturbation sizes, measured in the model's input space (clean vs adversarial).
        perturbation = (adversarial_samples - clean_samples).reshape(num_samples, -1)
        l2_per_sample = np.linalg.norm(perturbation, ord=2, axis=1)
        linf_per_sample = np.max(np.abs(perturbation), axis=1)

        # Relative perturbation = perturbation L2 norm / original image L2 norm.
        clean_l2 = np.linalg.norm(clean_samples.reshape(num_samples, -1), ord=2, axis=1)
        safe_clean_l2 = np.where(clean_l2 == 0, np.nan, clean_l2)
        relative_per_sample = l2_per_sample / safe_clean_l2

        # Confidence of each model's predicted class.
        clean_confidence = clean_probabilities[sample_indices, clean_predictions]
        adversarial_confidence = adversarial_probabilities[sample_indices, adversarial_predictions]

        # Top class flips among successful samples: original prediction -> adversarial prediction.
        flips = Counter(
            (int(clean_predictions[i]), int(adversarial_predictions[i]))
            for i in np.flatnonzero(success_mask)
        )
        top_class_flips = [
            {"from": src, "to": dst, "count": count}
            for (src, dst), count in flips.most_common(5)
        ]

        evaluation_results = {
            "num_samples": num_samples,
            "attack_success_rate": attack_success_rate,
            "l2_norm_min": float(np.min(l2_per_sample)),
            "l2_norm_max": float(np.max(l2_per_sample)),
            "average_l2_norm": float(np.mean(l2_per_sample)),
            "linf_norm_min": float(np.min(linf_per_sample)),
            "linf_norm_max": float(np.max(linf_per_sample)),
            "average_linf_norm": float(np.mean(linf_per_sample)),
            "average_relative_perturbation": float(np.nanmean(relative_per_sample)),
            "average_clean_confidence": float(np.mean(clean_confidence)),
            "average_adversarial_confidence": float(np.mean(adversarial_confidence)),
            "top_class_flips": top_class_flips,
        }

        visualization_context = self._build_visualization_context(
            clean_samples, adversarial_samples, true_labels, success_mask,
            clean_predictions, adversarial_predictions,
            l2_per_sample, relative_per_sample, clean_confidence, adversarial_confidence,
        )

        self._print_summary(evaluation_results)

        return evaluation_results, [], visualization_context

    @staticmethod
    def _build_visualization_context(clean_samples, adversarial_samples, true_labels,
                                     success_mask, clean_predictions, adversarial_predictions,
                                     l2_per_sample, relative_per_sample,
                                     clean_confidence, adversarial_confidence):
        # Illustrate the attack with a successful flip (fall back to the first sample).
        candidates = np.flatnonzero(success_mask)
        example_index = int(candidates[0]) if candidates.size else 0

        return DeepfoolEvasionVisualizationContext(
            base_accuracy=None,
            example_clean=clean_samples[example_index],
            example_adversarial=adversarial_samples[example_index],
            example_true_label=int(true_labels[example_index]),
            example_original_prediction=int(clean_predictions[example_index]),
            example_adversarial_prediction=int(adversarial_predictions[example_index]),
            example_l2_norm=float(l2_per_sample[example_index]),
            example_relative_perturbation=float(relative_per_sample[example_index]),
            example_clean_confidence=float(clean_confidence[example_index]),
            example_adversarial_confidence=float(adversarial_confidence[example_index]),
        )

    @staticmethod
    def _print_summary(results):
        print("DeepFool evasion evaluation summary:")
        print(f"  Samples evaluated:            {results['num_samples']}")
        print(f"  Attack success rate:          {results['attack_success_rate'] * 100:6.2f}%")
        print(f"  L2 norm range:                [{results['l2_norm_min']:.4f}, {results['l2_norm_max']:.4f}]")
        print(f"  Avg L2 norm:                  {results['average_l2_norm']:.4f}")
        print(f"  Linf norm range:              [{results['linf_norm_min']:.4f}, {results['linf_norm_max']:.4f}]")
        print(f"  Avg Linf norm:                {results['average_linf_norm']:.4f}")
        print(f"  Avg relative perturbation:    {results['average_relative_perturbation'] * 100:6.2f}%")
        print(f"  Avg clean confidence:         {results['average_clean_confidence'] * 100:6.2f}%")
        print(f"  Avg adversarial confidence:   {results['average_adversarial_confidence'] * 100:6.2f}%")
        if results["top_class_flips"]:
            print("  Top class flips:")
            for flip in results["top_class_flips"]:
                times = "time" if flip["count"] == 1 else "times"
                print(f"    {flip['from']} -> {flip['to']}: {flip['count']} {times}")
