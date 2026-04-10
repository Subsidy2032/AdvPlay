from pathlib import Path

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.evasion_context import EvasionContext
from advplay.model_ops.evaluators.base_evaluator import BaseEvaluator
from advplay import paths
from advplay.model_ops.model_loaders.base_model_loader import BaseModelLoader

class EvasionEvaluator(BaseAttackEvaluator, attack_type="evasion"):
    def evaluate(self, context: EvasionContext):
        training_framework = context.training_framework
        model_path = context.model_path
        samples_data = context.samples_data
        perturbed_samples = context.perturbed_samples
        target_label = context.target_label

        evaluator_cls = BaseEvaluator.registry.get(context.training_framework)
        evaluation_results = {}

        default_path = paths.MODELS
        if not Path(model_path).is_file():
            model_path = default_path / model_path
        loader_cls = BaseModelLoader.registry.get(training_framework)
        loader = loader_cls(model_path)
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

        return evaluation_results, []
