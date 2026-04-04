from sklearn.metrics import confusion_matrix

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.poisoning_context import PoisoningContext
from advplay.model_ops import registry

class PoisoningEvaluator(BaseAttackEvaluator, attack_type="poisoning"):
    def evaluate(self, context: PoisoningContext):
        X_test = context.X_test
        y_test = context.y_test

        base_model = registry.train(
            context.training_framework,
            context.model,
            context.clean_dataset["X_train"],
            context.clean_dataset["y_train"],
            config=context.training_configuration
        )

        base_acc = registry.evaluate_model_accuracy(context.training_framework, base_model, X_test, y_test)

        evaluation_results = {}
        evaluation_results["base_accuracy"] = base_acc
        evaluation_results["base_confusion_matrix"] = confusion_matrix(y_test, registry.predict(context.training_framework, base_model, X_test))
        evaluation_results["base_model"] = base_model
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
                                                                                                                          model, X_test)),
                                                            "poisoned_model": model})
            if acc < min_acc["acc"]:
                min_acc = {"acc": acc, "portion": portion, "model": model, "X_poisoned": data["X_train"], "y_poisoned": data["y_train"]}

        evaluation_results["min_accuracy"] = min_acc
        return evaluation_results
