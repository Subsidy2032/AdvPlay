from collections import Counter
from pathlib import Path
import numpy as np

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.evasion_evaluation_context import EvasionEvaluationContext
from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.visualization.contexts.elasticnet_evasion_visualization_context import ElasticNetEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


# ElasticNet trades L1 against L2 with a beta weight; under the 'EN' decision rule the
# elastic-net distortion of a perturbation is beta * ||delta||_1 + ||delta||_2^2. The
# evaluation context does not carry beta, so we use the attack's default (0.001).
ELASTIC_NET_BETA = 0.001

# Pixels whose perturbation is at or below this magnitude are treated as unchanged.
# ElasticNet's L1 term drives many perturbations to (near) exactly zero, so the
# sparsity measurement is not sensitive to the exact threshold.
SPARSITY_TOLERANCE = 1e-8

# Number of perturbations shown as heatmaps in the visualization.
MAX_HEATMAP_EXAMPLES = 10


class ElasticNetEvasionEvaluator(BaseAttackEvaluator,
                                 attack_type=available_attacks.EVASION,
                                 attack_subtype=evasion_techniques.ELASTICNET):
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

        clean_probabilities = evaluator.predict_proba(clean_samples)
        adversarial_probabilities = evaluator.predict_proba(adversarial_samples)
        clean_predictions = np.argmax(clean_probabilities, axis=1)
        adversarial_predictions = np.argmax(adversarial_probabilities, axis=1)

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
        l1_per_sample = np.sum(np.abs(perturbation), axis=1)
        l2_per_sample = np.linalg.norm(perturbation, ord=2, axis=1)
        linf_per_sample = np.max(np.abs(perturbation), axis=1)
        # Elastic-net distortion: beta * L1 + L2^2 (the 'EN' objective ElasticNet minimizes).
        elastic_per_sample = ELASTIC_NET_BETA * l1_per_sample + l2_per_sample ** 2

        # Sparsity = percentage of pixels the attack left unchanged.
        unchanged_mask = np.abs(perturbation) <= SPARSITY_TOLERANCE
        sparsity_per_sample = unchanged_mask.mean(axis=1) * 100.0

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
            "targeted": targeted,
            "target_label": int(target_labels[0]) if targeted else None,
            "num_samples": num_samples,
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "attack_success_rate": attack_success_rate,
            "average_l1_distortion": float(np.mean(l1_per_sample)),
            "average_l2_distortion": float(np.mean(l2_per_sample)),
            "average_linf_distortion": float(np.mean(linf_per_sample)),
            "average_elastic_distortion": float(np.mean(elastic_per_sample)),
            "average_sparsity": float(np.mean(sparsity_per_sample)),
            "top_class_flips": top_class_flips,
        }

        visualization_context = self._build_visualization_context(
            clean_samples, adversarial_samples, true_labels, target_labels, targeted,
            success_mask, clean_predictions, adversarial_predictions,
            l1_per_sample, l2_per_sample, linf_per_sample, elastic_per_sample,
            sparsity_per_sample,
        )

        self._print_summary(evaluation_results)

        return evaluation_results, [], visualization_context

    @staticmethod
    def _build_visualization_context(clean_samples, adversarial_samples, true_labels,
                                     target_labels, targeted, success_mask,
                                     clean_predictions, adversarial_predictions,
                                     l1_per_sample, l2_per_sample, linf_per_sample,
                                     elastic_per_sample, sparsity_per_sample):
        # Illustrate the attack with a successful example (fall back to the first sample).
        candidates = np.flatnonzero(success_mask)
        example_index = int(candidates[0]) if candidates.size else 0

        # Show heatmaps for the first handful of examples (kept aligned by index).
        heatmap_count = min(MAX_HEATMAP_EXAMPLES, len(clean_samples))
        heatmap_slice = slice(0, heatmap_count)

        return ElasticNetEvasionVisualizationContext(
            base_accuracy=None,
            targeted=targeted,
            example_clean=clean_samples[example_index],
            example_adversarial=adversarial_samples[example_index],
            example_true_label=int(true_labels[example_index]),
            example_target_label=int(target_labels[example_index]) if targeted else None,
            example_original_prediction=int(clean_predictions[example_index]),
            example_adversarial_prediction=int(adversarial_predictions[example_index]),
            example_l1_norm=float(l1_per_sample[example_index]),
            example_l2_norm=float(l2_per_sample[example_index]),
            example_linf_norm=float(linf_per_sample[example_index]),
            example_elastic_norm=float(elastic_per_sample[example_index]),
            l1_per_sample=l1_per_sample,
            l2_per_sample=l2_per_sample,
            linf_per_sample=linf_per_sample,
            elastic_per_sample=elastic_per_sample,
            sparsity_per_sample=sparsity_per_sample,
            heatmap_clean=np.asarray(clean_samples[heatmap_slice]),
            heatmap_adversarial=np.asarray(adversarial_samples[heatmap_slice]),
        )

    @staticmethod
    def _print_summary(results):
        print("ElasticNet evasion evaluation summary:")
        mode = "targeted" if results["targeted"] else "untargeted"
        print(f"  Mode: {mode}" + (f" (target = {results['target_label']})" if results["targeted"] else ""))
        print(f"  Samples evaluated:            {results['num_samples']}")
        print(f"  Clean accuracy:               {results['clean_accuracy'] * 100:6.2f}%")
        print(f"  Adversarial accuracy:         {results['adversarial_accuracy'] * 100:6.2f}%")
        print(f"  Attack success rate:          {results['attack_success_rate'] * 100:6.2f}%")
        print(f"  Avg L1 distortion:            {results['average_l1_distortion']:.4f}")
        print(f"  Avg L2 distortion:            {results['average_l2_distortion']:.4f}")
        print(f"  Avg Linf distortion:          {results['average_linf_distortion']:.4f}")
        print(f"  Avg elastic distortion:       {results['average_elastic_distortion']:.4f}")
        print(f"  Avg sparsity (unchanged):     {results['average_sparsity']:6.2f}%")
        if results["top_class_flips"]:
            print("  Top class flips:")
            for flip in results["top_class_flips"]:
                times = "time" if flip["count"] == 1 else "times"
                print(f"    {flip['from']} -> {flip['to']}: {flip['count']} {times}")
