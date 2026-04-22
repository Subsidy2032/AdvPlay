import numpy as np

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.clean_label_poisoning_evaluation_context import CleanLabelPoisoningEvaluationContext
from advplay.ml.ops.evaluators.base_evaluator import BaseEvaluator
from advplay.ml.ops.trainers.base_trainer import BaseTrainer
from advplay.variables import available_attacks, poisoning_techniques
from advplay import paths
from advplay.utils import load_files

class CleanLabelPoisoningEvaluator(BaseAttackEvaluator,
                                   attack_type=available_attacks.POISONING,
                                   attack_subtype=poisoning_techniques.CLEAN_LABEL):
    def evaluate(self, context: CleanLabelPoisoningEvaluationContext):
        training_framework = context.training_framework
        training_configuration = context.training_configuration
        if isinstance(training_configuration, str):
            default_path = paths.CONFIGS / training_framework
            training_configuration = load_files.load_json(default_path, training_configuration)

        X_clean = context.X_test
        y = np.asarray(context.y).ravel()
        X_poisoned = context.X_poisoned
        indices = np.asarray(context.indices_to_poison).ravel().astype(int)
        target_sample = context.target_sample
        target_label = context.target_label

        evaluator_cls = BaseEvaluator.registry.get(training_framework)
        trainer_cls = BaseTrainer.registry.get((training_framework, context.model))

        base_model = trainer_cls(X_clean, y, training_configuration).train()
        poisoned_model = trainer_cls(X_poisoned, y, training_configuration).train()

        base_evaluator = evaluator_cls(base_model)
        poisoned_evaluator = evaluator_cls(poisoned_model)

        base_accuracy = base_evaluator.accuracy(X_clean, y)
        poisoned_accuracy = poisoned_evaluator.accuracy(X_clean, y)

        base_predictions = base_evaluator.predict(X_poisoned[indices])
        poisoned_predictions = poisoned_evaluator.predict(X_poisoned[indices])
        true_labels = y[indices]

        base_misclassified = int(np.sum(base_predictions != true_labels))
        poisoned_misclassified = int(np.sum(poisoned_predictions != true_labels))

        n_poisoned = int(len(indices))
        base_percentage = (base_misclassified / n_poisoned) * 100 if n_poisoned else 0.0
        poisoned_percentage = (poisoned_misclassified / n_poisoned) * 100 if n_poisoned else 0.0

        target_base_prediction = int(base_evaluator.predict(target_sample)[0])
        target_poisoned_prediction = int(poisoned_evaluator.predict(target_sample)[0])
        target_attack_succeeded = (target_label is not None
                                   and target_base_prediction == target_label
                                   and target_poisoned_prediction != target_label)

        evaluation_results = {
            "base_accuracy": base_accuracy,
            "poisoned_accuracy": poisoned_accuracy,
            "accuracy_reduction": base_accuracy - poisoned_accuracy,
            "n_samples_poisoned": n_poisoned,
            "base_misclassified_at_indices": base_misclassified,
            "base_misclassification_percentage": base_percentage,
            "poisoned_misclassified_at_indices": poisoned_misclassified,
            "poisoned_misclassification_percentage": poisoned_percentage,
            "target_label": target_label,
            "target_base_prediction": target_base_prediction,
            "target_poisoned_prediction": target_poisoned_prediction,
            "target_attack_succeeded": bool(target_attack_succeeded),
        }

        print("Clean-label poisoning evaluation summary:")
        print(f"Base accuracy: {base_accuracy:.4f}")
        print(f"Poisoned accuracy: {poisoned_accuracy:.4f}")
        print(f"Accuracy reduction: {base_accuracy - poisoned_accuracy:.4f}")
        print(f"Misclassifications at poisoned indices: "
              f"base {base_misclassified}/{n_poisoned} ({base_percentage:.1f}%) → "
              f"poisoned {poisoned_misclassified}/{n_poisoned} ({poisoned_percentage:.1f}%)")
        print(f"Target sample (true label = {target_label}): "
              f"base predicts {target_base_prediction}, poisoned predicts {target_poisoned_prediction} "
              f"→ attack {'SUCCEEDED' if target_attack_succeeded else 'FAILED'}")

        models = [
            (training_framework, base_model, f"{context.model_name}_clean"),
            (training_framework, poisoned_model, f"{context.model_name}_poisoned"),
        ]

        return evaluation_results, models, None
