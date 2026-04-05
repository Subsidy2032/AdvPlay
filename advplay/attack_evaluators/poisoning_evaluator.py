from sklearn.metrics import confusion_matrix

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.poisoning_context import PoisoningContext
from advplay.model_ops import registry

class PoisoningEvaluator(BaseAttackEvaluator, attack_type="poisoning"):
    def evaluate(self, context: PoisoningContext):
        X_test = context.X_test
        y_test = context.y_test

        models = []

        base_model = registry.train(
            context.training_framework,
            context.model,
            context.clean_dataset["X_train"],
            context.clean_dataset["y_train"],
            config=context.training_configuration
        )

        models.append((context.training_framework, base_model, context.model_name))

        base_acc = registry.evaluate_model_accuracy(context.training_framework, base_model, X_test, y_test)

        evaluation_results = {}
        evaluation_results["base_accuracy"] = base_acc
        evaluation_results["base_confusion_matrix"] = confusion_matrix(y_test, registry.predict(context.training_framework, base_model, X_test))
        evaluation_results["poisoning_results"] = []

        min_acc = {"acc": 1.1, "portion": None, "model": None, "X_poisoned": None, "y_poisoned": None}
        for portion, data in context.poisoned_datasets.items():
            model = registry.train(
                context.training_framework,
                context.model,
                data["X_train"],
                data["y_train"],
                config=context.training_configuration
            )

            acc = registry.evaluate_model_accuracy(context.training_framework, model, X_test, y_test)

            evaluation_results["poisoning_results"].append({"portion": portion,
                                                            "n_samples_poisoned": data["n_samples_poisoned"],
                                                            "accuracy": acc,
                                                            "confusion_matrix": confusion_matrix(y_test, registry.predict(context.training_framework, 
                                                                                                                          model, X_test))})
            if acc < min_acc["acc"]:
                min_acc = {"acc": acc, "portion": portion}
                min_acc_model = model

            models.append((context.training_framework, min_acc_model, f"{context.model_name}_poisoned"))

        evaluation_results["min_accuracy"] = min_acc
        return evaluation_results, models
