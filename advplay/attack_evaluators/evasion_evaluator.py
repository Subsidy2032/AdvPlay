from sklearn.metrics import confusion_matrix

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.evasion_context import EvasionContext
from advplay.model_ops import registry

class EvasionEvaluator(BaseAttackEvaluator, attack_type="evasion"):
    def evaluate(self, context: EvasionContext):
        training_framework = context.training_framework
        model_path = context.model_path
        samples_data = context.samples_data
        perturbed_samples = context.perturbed_samples
        target_label = context.target_label

        evaluation_results = {}
        model = registry.load_model(training_framework, model_path)
        original_predictions = registry.predict(training_framework, model, samples_data)
        perturbed_predictions = registry.predict(training_framework, model, perturbed_samples)
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
        if self.target_label is not None:
            num_target_mispredictions = sum(original != 2 and perturbed == 2 for original, perturbed in zip(original_predictions, perturbed_predictions))
            percentage_target_mispredictions = (num_target_mispredictions / num_samples) * 100
            evaluation_results["num_target_mispredictions"] = num_target_mispredictions
            evaluation_results["percentage_target_mispredictions"] = percentage_target_mispredictions

        return evaluation_results, []
