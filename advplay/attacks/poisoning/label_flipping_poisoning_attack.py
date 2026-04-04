import numpy as np
import os
from sklearn.model_selection import train_test_split

from advplay.utils import save_model
from advplay import paths
from advplay.loggers.json_logger import JsonLogger
from advplay.attacks.poisoning.poisoing_attack import PoisoningAttack
from advplay.variables import available_attacks, poisoning_techniques
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.attack_evaluators.poisoning_evaluator import PoisoningEvaluator
from advplay.attack_evaluators.contexts.poisoning_context import PoisoningContext
from advplay.model_ops import registry

class LabelFlippingPoisoningAttack(PoisoningAttack,
                                    attack_type=available_attacks.POISONING,
                                    attack_subtype=poisoning_techniques.LABEL_FLIPPING):
    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

    def execute(self):
        super().execute()

        X = np.delete(self.dataset.data, self.label_column, axis=1) if self.dataset else self.features_dataset.data
        y_raw = self.dataset.data[:, self.label_column] if self.dataset else self.labels_array.data
        labels_unique = np.unique(y_raw)
        label_map = {lbl: i for i, lbl in enumerate(labels_unique)}
        reverse_label_map = {i: lbl for lbl, i in label_map.items()}

        y = np.vectorize(label_map.get)(y_raw).astype(int)

        if len(np.unique(y)) <= 1:
            raise ValueError("Poisoning requires at least two classes")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_portion, random_state=self.seed)

        source_mask = (
            (y_train == label_map[self.source]) if self.source else
            (y_train != label_map[self.target]) if self.target else
            np.ones_like(y_train, bool)
        )
        y_source = y_train[source_mask]
        n_samples = len(y_source)

        poisoned_datasets = {}
        steps = max(1, int((self.max_portion_to_poison - self.min_portion_to_poison) / self.step) + 1)
        rng = np.random.default_rng(self.seed)

        for portion in np.linspace(self.min_portion_to_poison, self.max_portion_to_poison, steps):
            n_to_poison = int(n_samples * portion)
            idx = rng.choice(np.where(source_mask)[0], size=n_to_poison, replace=False)

            y_poisoned = y_train.copy() if self.override else np.concatenate([y_train, y_train[idx]])
            y_poisoned[idx] = self.poison(y_train[idx], np.unique(y), label_map, rng)

            X_poisoned = X_train.copy() if self.override else np.vstack([X_train, X_train[idx]])

            poisoned_datasets[portion] = {
                "X_train": X_poisoned,
                "y_train": y_poisoned,
                "n_samples_poisoned": int(n_samples * portion)
            }

        clean_dataset = {
            "X_train": X_train,
            "y_train": y_train
        }
        poisoning_context = PoisoningContext(self.model, X_test, y_test, self.training_framework, self.training_configuration, clean_dataset, poisoned_datasets)
        poisoning_evaluator = PoisoningEvaluator()
        results = poisoning_evaluator.evaluate(poisoning_context)

        try:
            save_model.save_model(self.training_framework, results["base_model"], self.model_name)
            save_model.save_model(self.training_framework, results["min_accuracy"]["model"], f"{self.model_name}_poisoned")

            X_poisoned = results["min_accuracy"]["X_poisoned"]
            y_poisoned = results["min_accuracy"]["y_poisoned"]
            y_poisoned_original = np.vectorize(reverse_label_map.get)(y_poisoned)

            self.save_dataset(X_poisoned, y_poisoned_original)

        except Exception as e:
            raise RuntimeError(f"Failed to save model(s) or dataset: {e}")

        sanitized_results = self.sanitize_results_for_logging(results)
        self.log_attack_results(labels_unique, sanitized_results, self.log_file_path)

    def sanitize_results_for_logging(self, results):
        sanitized = {
            "base_accuracy": results["base_accuracy"],
            "base_confusion_matrix": results["base_confusion_matrix"],
            "min_accuracy": {
                "acc": results["min_accuracy"]["acc"],
                "portion": results["min_accuracy"]["portion"]
            },
            "poisoning_results": []
        }

        for entry in results["poisoning_results"]:
            sanitized["poisoning_results"].append({
                "portion": entry["portion"],
                "n_samples_poisoned": entry["n_samples_poisoned"],
                "accuracy": entry["accuracy"],
                "confusion_matrix": entry["confusion_matrix"]
            })

        return sanitized

    def poison(self, labels_to_poison, labels, label_map, rng):
        if self.target is not None:
            return np.full_like(labels_to_poison, fill_value=label_map[self.target])
        poisoned = labels_to_poison.copy()
        for i in range(len(poisoned)):
            poisoned[i] = rng.choice(labels[labels != poisoned[i]])
        return poisoned

    def save_dataset(self, X_poisoned, y_poisoned):
        if self.split:
            X_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.features_dataset_name}_poisoned"
            y_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.labels_dataset_name}_poisoned"
            os.makedirs(X_dataset_path.parent, exist_ok=True)
            os.makedirs(y_dataset_path.parent, exist_ok=True)

            X_loaded_dataset = LoadedDataset(
                X_poisoned,
                self.X_source_type,
                self.X_metadata
            )

            y_loaded_dataset = LoadedDataset(
                y_poisoned,
                self.y_source_type,
                self.y_metadata
            )
            registry.save_dataset(X_loaded_dataset, X_dataset_path)
            registry.save_dataset(y_loaded_dataset, y_dataset_path)

            print(f"Features dataset saved to {X_dataset_path}.{self.X_source_type}")
            print(f"Labels dataset saved to {y_dataset_path}.{self.y_source_type}")

        else:
            dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.dataset_name}_poisoned"
            os.makedirs(dataset_path.parent, exist_ok=True)

            poisoned_dataset = np.insert(X_poisoned, self.label_column, y_poisoned, axis=1)
            loaded_dataset = LoadedDataset(poisoned_dataset,
                                           self.source_type, self.metadata)
            registry.save_dataset(loaded_dataset, dataset_path)

            print(f"Dataset saved to {dataset_path}.{self.source_type}")

    def log_attack_results(self, labels, results, log_file_path):
        log_entry = {
            "attack": self.attack_type, "technique": self.attack_subtype,
            "training_framework": self.training_framework, "model": self.model,
            "training_configuration": self.training_configuration,
            "test_portion": self.test_portion, "min_portion_to_poison": self.min_portion_to_poison,
            "max_portion_to_poison": self.max_portion_to_poison, "source": self.source, "target": self.target,
            "override": self.override, "seed": self.seed, "step": self.step, "model_name": self.model_name,
            "labels": labels, "base_accuracy": results["base_accuracy"],
            "most_effective_portion": results["min_accuracy"]["portion"], "min_accuracy": results["min_accuracy"]["acc"],
            "base_confusion_matrix": results["base_confusion_matrix"], "poisoning_results": results["poisoning_results"]
        }
        logger = JsonLogger(log_file_path)
        logger.log(log_entry)
