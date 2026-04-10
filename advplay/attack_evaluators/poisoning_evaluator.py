from sklearn.metrics import confusion_matrix

from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.attack_evaluators.contexts.poisoning_context import PoisoningContext
from advplay.model_ops.evaluators.base_evaluator import BaseEvaluator
from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay import paths
from advplay.utils import load_files

class PoisoningEvaluator(BaseAttackEvaluator, attack_type="poisoning"):
    def evaluate(self, context: PoisoningContext):
        evaluator_cls = BaseEvaluator.registry.get(context.training_framework)
        X_test = context.X_test
        y_test = context.y_test

        models = []

        default_path = paths.CONFIGS / context.training_framework
        if isinstance(context.training_configuration, str):
            context.training_configuration = load_files.load_json(default_path, context.training_configuration)
        trainer_cls = BaseTrainer.registry.get((context.training_framework, context.model))
        trainer = trainer_cls(context.clean_dataset["X_train"], context.clean_dataset["y_train"], context.training_configuration)
        base_model = trainer.train()

        evaluator = evaluator_cls(base_model)

        models.append((context.training_framework, base_model, context.model_name))

        base_acc = evaluator.accuracy(X_test, y_test)

        evaluation_results = {}
        evaluation_results["base_accuracy"] = base_acc
        evaluation_results["base_confusion_matrix"] = confusion_matrix(y_test, evaluator.predict(X_test))
        evaluation_results["poisoning_results"] = []

        min_acc = {"acc": 1.1, "portion": None, "model": None, "X_poisoned": None, "y_poisoned": None}
        for portion, data in context.poisoned_datasets.items():
            default_path = paths.CONFIGS / context.training_framework
            if isinstance(context.training_configuration, str):
                context.training_configuration = load_files.load_json(default_path, context.training_configuration)
            trainer_cls = BaseTrainer.registry.get((context.training_framework, context.model))
            trainer = trainer_cls(data["X_train"], data["y_train"], context.training_configuration)
            model = trainer.train()

            evaluator = evaluator_cls(model)
            acc = evaluator.accuracy(X_test, y_test)
            evaluation_results["poisoning_results"].append({"portion": portion,
                                                            "n_samples_poisoned": data["n_samples_poisoned"],
                                                            "accuracy": acc,
                                                            "confusion_matrix": confusion_matrix(y_test, evaluator.predict(X_test))})
            if acc < min_acc["acc"]:
                min_acc = {"acc": acc, "portion": portion}
                min_acc_model = model

        models.append((context.training_framework, min_acc_model, f"{context.model_name}_poisoned"))
        evaluation_results["min_accuracy"] = min_acc

        print("Evaluation results summary:")
        print(f"Base accuracy: {base_acc:.2f}")
        print(f"Lowest poisoned accuracy: {min_acc['acc']:.2f} ({min_acc['portion'] * 100:.1f}% poisoned)")
        print(f"Accuracy reduction: {base_acc - min_acc['acc']:.2f}\n")
        return evaluation_results, models
