import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from advplay.model_ops import registry
from advplay.utils import save_model
from advplay import paths
from advplay.utils.append_log_entry import append_log_entry
from advplay.attacks.poisoning.poisoing_attack import PoisoningAttack
from advplay.variables import available_attacks, poisoning_techniques
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset

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
        y = np.vectorize(label_map.get)(y_raw).astype(int)

        if len(np.unique(y)) <= 1:
            raise ValueError("Poisoning requires at least two classes")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_portion, random_state=self.seed)

        base_model = registry.train(self.training_framework, self.model, X_train, y_train)

        source_mask = (
            (y_train == label_map[self.source]) if self.source else
            (y_train != label_map[self.target]) if self.target else
            np.ones_like(y_train, bool)
        )
        y_source = y_train[source_mask]
        n_samples = len(y_source)

        poisoned_models, poisoned_datasets = {}, {}
        steps = max(1, int((self.max_portion_to_poison - self.min_portion_to_poison) / self.step) + 1)
        rng = np.random.default_rng(self.seed)

        for portion in np.linspace(self.min_portion_to_poison, self.max_portion_to_poison, steps):
            n_to_poison = int(n_samples * portion)
            idx = rng.choice(np.where(source_mask)[0], size=n_to_poison, replace=False)
            y_poisoned = y_train.copy() if self.override else np.concatenate([y_train, y_train[idx]])
            y_poisoned[idx] = self.poison(y_train[idx], np.unique(y), label_map, rng)

            X_poisoned = X_train.copy() if self.override else np.vstack([X_train, X_train[idx]])
            poisoned_models[portion] = registry.train(self.training_framework, self.model, X_poisoned, y_poisoned)
            poisoned_datasets[portion] = np.column_stack((X_poisoned, y_poisoned))

        if not poisoned_models:
            raise RuntimeError("No poisoned models generated; check configuration")

        results = self.evaluate(n_samples, X_test, y_test, labels_unique, base_model, poisoned_models)
        print(f"Most effective portion: {results['most_effective_portion']*100:.1f}%\n")

        try:
            save_model.save_model(self.training_framework, base_model, self.model_name)
            save_model.save_model(self.training_framework, poisoned_models[results["most_effective_portion"]],
                                  f"{self.model_name}_poisoned")

            if self.split:
                X_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.model_name}_dataset_X"
                y_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.model_name}_dataset_y"
                os.makedirs(X_dataset_path.parent, exist_ok=True)
                os.makedirs(y_dataset_path.parent, exist_ok=True)
                X_train_poisoned_final = poisoned_datasets[results["most_effective_portion"]][:, :-1]
                y_train_poisoned_final = poisoned_datasets[results["most_effective_portion"]][:, -1]

                X_loaded_dataset = LoadedDataset(
                    X_train_poisoned_final,
                    self.X_source_type,
                    self.X_metadata
                )

                y_loaded_dataset = LoadedDataset(
                    y_train_poisoned_final,
                    self.y_source_type,
                    self.y_metadata
                )
                registry.save_dataset(X_loaded_dataset, X_dataset_path)
                registry.save_dataset(y_loaded_dataset, y_dataset_path)

            else:
                dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.model_name}_dataset"
                os.makedirs(dataset_path.parent, exist_ok=True)
                loaded_dataset = LoadedDataset(poisoned_datasets[results["most_effective_portion"]],
                                               self.source_type, self.metadata)
                registry.save_dataset(loaded_dataset, dataset_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save model(s) or dataset: {e}")

        self.log_attack_results(labels_unique, results, self.log_file_path)

    def poison(self, labels_to_poison, labels, label_map, rng):
        if self.target is not None:
            return np.full_like(labels_to_poison, fill_value=label_map[self.target])
        poisoned = labels_to_poison.copy()
        for i in range(len(poisoned)):
            poisoned[i] = rng.choice(labels[labels != poisoned[i]])
        return poisoned

    def evaluate(self, n_samples, X_test, y_test, labels_unique, base_model, poisoned_models):
        base_acc = registry.evaluate_model_accuracy(self.training_framework, base_model, X_test, y_test)
        print(f"Base model accuracy: {base_acc:.2f}\n")
        min_acc, best_portion = 1.1, None
        evaluation_results = {}

        evaluation_results["base_accuracy"] = base_acc
        evaluation_results["base_confusion_matrix"] = confusion_matrix(y_test, base_model.predict(X_test))
        evaluation_results["poisoning_results"] = []

        for portion, model in poisoned_models.items():
            acc = registry.evaluate_model_accuracy(self.training_framework, model, X_test, y_test)
            print(f"Model with {int(n_samples*portion)} poisoned samples ({portion*100:.1f}%): accuracy={acc:.2f}, "
                  f"attack success={base_acc-acc:.2f}")
            evaluation_results["poisoning_results"].append({"portion": portion,
                                 "n_samples_poisoned": int(n_samples*portion),
                                 "accuracy": acc,
                                 "confusion_matrix": confusion_matrix(y_test, model.predict(X_test))})
            if acc < min_acc:
                min_acc, best_portion = acc, portion

        evaluation_results["most_effective_portion"] = best_portion
        evaluation_results["min_accuracy"] = min_acc
        return evaluation_results

    def log_attack_results(self, labels, results, log_file_path):
        log_entry = {
            "attack": self.attack_type, "technique": self.technique,
            "training_framework": self.training_framework, "model": self.model,
            "training_configuration": self.training_configuration,
            "test_portion": self.test_portion, "min_portion_to_poison": self.min_portion_to_poison,
            "max_portion_to_poison": self.max_portion_to_poison, "source": self.source, "target": self.target,
            "override": self.override, "seed": self.seed, "step": self.step, "model_name": self.model_name,
            "labels": labels, "base_accuracy": results["base_accuracy"],
            "most_effective_portion": results["most_effective_portion"], "min_accuracy": results["min_accuracy"],
            "base_confusion_matrix": results["base_confusion_matrix"], "poisoning_results": results["poisoning_results"]
        }
        append_log_entry(log_file_path, log_entry)
