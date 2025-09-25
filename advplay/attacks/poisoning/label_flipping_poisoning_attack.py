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

class LabelFlippingPoisoningAttack(PoisoningAttack, attack_type=available_attacks.POISONING, attack_subtype=poisoning_techniques.LABEL_FLIPPING):
    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

    def execute(self):
        super().execute()

        if self.dataset is not None:
            try:
                X = np.delete(self.dataset.data, self.label_column, axis=1)
                y = self.dataset.data[:, self.label_column]
            except Exception as e:
                raise RuntimeError(f"Failed to split features and labels: {e}")

        else:
            X = self.features_dataset.data
            y = self.labels_array.data

        unique_labels_original = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels_original)}
        mapped_labels = np.vectorize(label_map.get)(y)
        y = mapped_labels.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_portion,
                                                                random_state=self.seed)

        unique_labels = np.unique(y)

        if len(unique_labels) <= 1:
            raise ValueError("Only one class is present; poisoning requires at least two distinct classes")

        base_model = registry.train(self.training_framework, self.training_algorithm, X_train, y_train)

        rng = np.random.default_rng(self.seed)
        poisoned_models_dict = {}
        poisoned_datasets_dict = {}

        num_steps = 1

        if self.step > 0:
            num_steps = int((self.max_portion_to_poison - self.min_portion_to_poison) / self.step) + 1
        if num_steps <= 0:
            raise ValueError("Number of poisoning steps must be positive; check min/max_portion and step values")


        if self.source is not None:
            source_mask = y_train == label_map[self.source]

        elif self.target is not None:
            source_mask = y_train != label_map[self.target]

        else:
            source_mask = np.ones_like(y_train, dtype=bool)

        y_source = y_train[source_mask]
        n_samples = len(y_source)

        for portion_to_poison in np.linspace(self.min_portion_to_poison, self.max_portion_to_poison, num_steps):
            n_to_poison = int(n_samples * portion_to_poison)
            print(f"Poisoning {n_to_poison} samples ({portion_to_poison * 100:.1f}%) from the dataset")

            try:
                source_indices = np.where(source_mask)[0]
                indices_to_flip = rng.choice(source_indices, size=n_to_poison, replace=False)
                poisoned_labels = self.poison(y_train[indices_to_flip].copy(), unique_labels, label_map, rng)
                poisoned_labels = poisoned_labels.astype(int)
            except Exception as e:
                raise RuntimeError(f"Failed during poisoning step at {portion_to_poison * 100:.1f}%: {e}")

            if self.override:
                X_train_poisoned = X_train.copy()
                y_train_poisoned = y_train.copy()
                y_train_poisoned[indices_to_flip] = poisoned_labels

            else:
                X_train_poisoned = np.vstack([X_train, X_train[indices_to_flip]])
                y_train_poisoned = np.concatenate([y_train, poisoned_labels])

            y_train_poisoned = y_train_poisoned.astype(int)

            try:
                model = registry.train(self.training_framework, self.training_algorithm, X_train_poisoned, y_train_poisoned)
            except Exception as e:
                raise RuntimeError(f"Failed to train poisoned model at {portion_to_poison * 100:.1f}%: {e}")

            poisoned_models_dict[portion_to_poison] = model
            poisoned_datasets_dict[portion_to_poison] = np.column_stack((X_train_poisoned, y_train_poisoned))

        if not poisoned_models_dict:
            raise RuntimeError("No poisoned models were generated; check your configuration")

        evaluation_results = self.evaluate(n_samples, X_test, y_test, unique_labels_original, base_model, poisoned_models_dict)
        most_effective_portion = evaluation_results["most_effective_portion"]

        print(f"The attack was most effective when poisoning {most_effective_portion * 100:.1f}% of the training dataset\n")

        try:
            print(f"Saving base model\n")
            save_model.save_model(self.training_framework, base_model, self.model_name)

            print(f"Saving poisoned model and poisoned dataset\n")
            save_model.save_model(self.training_framework, poisoned_models_dict[most_effective_portion], f"{self.model_name}_poisoned")

            if self.split:
                X_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.model_name}_dataset_X"
                y_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.model_name}_dataset_y"

                os.makedirs(X_dataset_path.parent, exist_ok=True)
                os.makedirs(y_dataset_path.parent, exist_ok=True)

                X_train_poisoned_final = poisoned_datasets_dict[most_effective_portion][:, :-1]
                y_train_poisoned_final = poisoned_datasets_dict[most_effective_portion][:, -1]

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

                loaded_dataset = LoadedDataset(poisoned_datasets_dict[most_effective_portion],
                                               self.source_type, self.metadata)
                registry.save_dataset(loaded_dataset, dataset_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save model(s) and dataset: {e}")

        self.log_attack_results(unique_labels_original, evaluation_results, self.log_file_path)

    def poison(self, labels_to_poison, labels, label_map, rng):
        if self.target is not None:
            return np.full_like(labels_to_poison, fill_value=label_map[self.target])

        poisoned = labels_to_poison.copy()
        for i in range(len(poisoned)):
            labels_without_source = labels[labels != poisoned[i]]
            poisoned[i] = rng.choice(labels_without_source)

        return poisoned

    def evaluate(self, n_samples, X_test, y_test, unique_labels, base_model, poisoned_models: dict):
        min_accuracy = 1.1
        most_effective_portion = None
        evaluation_results = {}

        print("\n")
        base_model_accuracy = registry.evaluate_model_accuracy(self.training_framework, base_model, X_test, y_test)
        evaluation_results["base_accuracy"] = base_model_accuracy
        print(f"Base model accuracy is: {base_model_accuracy:.2f}\n")

        base_model_predictions = base_model.predict(X_test)
        base_confusion_mat = confusion_matrix(y_test, base_model_predictions)
        evaluation_results["base_confusion_matrix"] = base_confusion_mat

        evaluation_results["poisoning_results"] = []
        for portion_to_poison, model in poisoned_models.items():
            try:
                accuracy = registry.evaluate_model_accuracy(self.training_framework, model, X_test, y_test)
                n_to_poison = int(n_samples * portion_to_poison)
                print(f"The accuracy for the model with {n_to_poison} "
                      f"samples poisoned ({portion_to_poison * 100:.1f}%) is {accuracy:.2f}")

                print(f"Attack success rate: {base_model_accuracy - accuracy:.2f}")

                predictions = model.predict(X_test)
                confusion_mat = confusion_matrix(y_test, predictions)

                evaluation_results["poisoning_results"].append({
                    "portion_to_poison": portion_to_poison,
                    "n_samples_poisoned": n_to_poison,
                    "accuracy": accuracy,
                    "confusion_matrix": confusion_mat
                })

                if accuracy < min_accuracy:
                    min_accuracy = accuracy
                    most_effective_portion = portion_to_poison
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate poisoned model at {portion_to_poison * 100:.1f}%: {e}")

        evaluation_results["most_effective_portion"] = most_effective_portion
        evaluation_results["min_accuracy"] = min_accuracy
        return evaluation_results

    def log_attack_results(self, unique_labels, evaluation_results, log_file_path):
        log_entry = {
            "attack": self.attack_type,
            "technique": self.technique,
            "training_framework": self.training_framework,
            "training_algorithm": self.training_algorithm,
            "training_configuration": self.training_configuration,
            "test_portion": self.test_portion,
            "min_portion_to_poison": self.min_portion_to_poison,
            "max_portion_to_poison": self.max_portion_to_poison,
            "source": self.source,
            "target": self.target,
            "override": self.override,
            "seed": self.seed,
            "step": self.step,
            "model_name": self.model_name,
            "labels": unique_labels,
            "base_accuracy": evaluation_results["base_accuracy"],
            "base_confusion_matrix": evaluation_results["base_confusion_matrix"],
            "most_effective_portion": evaluation_results["most_effective_portion"],
            "min_accuracy": evaluation_results["min_accuracy"],
            "poisoning_results": evaluation_results["poisoning_results"]
        }

        append_log_entry(log_file_path, log_entry)
