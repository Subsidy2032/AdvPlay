from pathlib import Path
import numpy as np

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.evasion_evaluation_context import EvasionEvaluationContext
from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.visualization.contexts.fgsm_evasion_visualization_context import FGSMEvasionVisualizationContext
from advplay.variables import available_attacks, evasion_techniques
from advplay import paths


class FGSMEvasionEvaluator(BaseAttackEvaluator,
                           attack_type=available_attacks.EVASION,
                           attack_subtype=evasion_techniques.FGSM):
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
        sample_indices = np.arange(num_samples)

        clean_accuracy = float(np.mean(clean_predictions == true_labels))
        adversarial_accuracy = float(np.mean(adversarial_predictions == true_labels))

        if targeted:
            # Success = the model now predicts the attacker-chosen target class.
            attack_success_rate = float(np.mean(adversarial_predictions == target_labels))
        else:
            # Success = a sample the model originally classified correctly is now wrong.
            originally_correct = clean_predictions == true_labels
            if originally_correct.any():
                flipped = adversarial_predictions[originally_correct] != true_labels[originally_correct]
                attack_success_rate = float(np.mean(flipped))
            else:
                attack_success_rate = 0.0

        # Confidence of the predicted class for each sample.
        clean_confidence = clean_probabilities[sample_indices, clean_predictions]
        adversarial_confidence = adversarial_probabilities[sample_indices, adversarial_predictions]
        # Drop in the probability the model assigned to its original (clean) prediction.
        confidence_in_clean_class_after_attack = adversarial_probabilities[sample_indices, clean_predictions]
        confidence_drop = clean_confidence - confidence_in_clean_class_after_attack

        average_clean_confidence = float(np.mean(clean_confidence))
        average_adversarial_confidence = float(np.mean(adversarial_confidence))
        average_confidence_drop = float(np.mean(confidence_drop))

        # Perturbation sizes are measured in the model's input space (clean vs adversarial).
        perturbation = (adversarial_samples - clean_samples).reshape(num_samples, -1)
        l2_per_sample = np.linalg.norm(perturbation, ord=2, axis=1)
        linf_per_sample = np.max(np.abs(perturbation), axis=1)
        average_l2_perturbation = float(np.mean(l2_per_sample))
        max_linf_perturbation = float(np.max(linf_per_sample))

        evaluation_results = {
            "targeted": targeted,
            "target_label": int(target_labels[0]) if targeted else None,
            "num_samples": num_samples,
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "attack_success_rate": attack_success_rate,
            "average_clean_confidence": average_clean_confidence,
            "average_adversarial_confidence": average_adversarial_confidence,
            "average_confidence_drop": average_confidence_drop,
            "average_l2_perturbation": average_l2_perturbation,
            "max_linf_perturbation": max_linf_perturbation,
        }

        visualization_context = self._build_visualization_context(
            clean_samples, adversarial_samples, true_labels, target_labels, targeted,
            clean_probabilities, adversarial_probabilities,
            clean_predictions, adversarial_predictions,
        )

        self._print_summary(evaluation_results)

        return evaluation_results, [], visualization_context

    @staticmethod
    def _build_visualization_context(clean_samples, adversarial_samples, true_labels,
                                     target_labels, targeted, clean_probabilities,
                                     adversarial_probabilities, clean_predictions,
                                     adversarial_predictions):
        # Pick a sample that best illustrates the attack: a successful misclassification.
        if targeted:
            success_mask = adversarial_predictions == target_labels
        else:
            success_mask = (clean_predictions == true_labels) & (adversarial_predictions != true_labels)
        candidates = np.flatnonzero(success_mask)
        example_index = int(candidates[0]) if candidates.size else 0

        return FGSMEvasionVisualizationContext(
            base_accuracy=None,
            targeted=targeted,
            example_clean=clean_samples[example_index],
            example_adversarial=adversarial_samples[example_index],
            example_true_label=int(true_labels[example_index]),
            example_target_label=int(target_labels[example_index]) if targeted else None,
            example_clean_prediction=int(clean_predictions[example_index]),
            example_adversarial_prediction=int(adversarial_predictions[example_index]),
            example_clean_probabilities=clean_probabilities[example_index].tolist(),
            example_adversarial_probabilities=adversarial_probabilities[example_index].tolist(),
        )

    @staticmethod
    def _print_summary(results):
        print("FGSM evasion evaluation summary:")
        mode = "targeted" if results["targeted"] else "untargeted"
        print(f"  Mode: {mode}" + (f" (target = {results['target_label']})" if results["targeted"] else ""))
        print(f"  Samples evaluated:            {results['num_samples']}")
        print(f"  Clean accuracy:               {results['clean_accuracy'] * 100:6.2f}%")
        print(f"  Adversarial accuracy:         {results['adversarial_accuracy'] * 100:6.2f}%")
        print(f"  Attack success rate:          {results['attack_success_rate'] * 100:6.2f}%")
        print(f"  Avg clean confidence:         {results['average_clean_confidence'] * 100:6.2f}%")
        print(f"  Avg adversarial confidence:   {results['average_adversarial_confidence'] * 100:6.2f}%")
        print(f"  Avg confidence drop:          {results['average_confidence_drop'] * 100:6.2f}%")
        print(f"  Avg L2 perturbation:          {results['average_l2_perturbation']:.4f}")
        print(f"  Max Linf perturbation:        {results['max_linf_perturbation']:.4f}")
