from pathlib import Path

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.evasion_evaluation_context import EvasionEvaluationContext
from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay import paths
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.variables import available_attacks

class EvasionEvaluator(BaseAttackEvaluator, attack_type=available_attacks.EVASION, attack_subtype=None):
    def evaluate(self, context: EvasionEvaluationContext):
        model_name = context.model
        training_configuration = context.training_configuration
        training_framework = context.training_framework
        model_path = context.model_path
        samples_data = context.samples_data
        perturbed_samples = context.perturbed_samples
        target_label = context.target_label

        evaluator_cls = BaseEvaluator.get(context.training_framework, context.model)
        evaluation_results = {}

        default_path = paths.MODELS
        if not Path(model_path).is_file():
            model_path = default_path / model_path
        loader_cls = BaseModelLoader.registry.get(training_framework)
        loader = loader_cls(model_path, model_name, training_configuration)
        model = loader.load()
        evaluator = evaluator_cls(model)
        original_predictions = evaluator.predict(samples_data)
        perturbed_predictions = evaluator.predict(perturbed_samples)
        evaluation_results["original_predictions"] = original_predictions
        evaluation_results["perturbed_predictions"] = perturbed_predictions

        num_mispredictions = sum(original != perturbed for original, perturbed in zip(original_predictions, perturbed_predictions))
        num_samples = len(original_predictions)
        percentage_mispredicted = (num_mispredictions / num_samples) * 100
        evaluation_results["num_mispredictions"] = num_mispredictions
        evaluation_results["num_samples"] = num_samples
        evaluation_results["percentage_mispredicted"] = percentage_mispredicted

        evaluation_results["num_target_mispredictions"] = None
        evaluation_results["percentage_target_mispredictions"] = None
        if target_label is not None:
            num_target_mispredictions = sum(original != 2 and perturbed == 2 for original, perturbed in zip(original_predictions, perturbed_predictions))
            percentage_target_mispredictions = (num_target_mispredictions / num_samples) * 100
            evaluation_results["num_target_mispredictions"] = num_target_mispredictions
            evaluation_results["percentage_target_mispredictions"] = percentage_target_mispredictions

        return evaluation_results, [], None
