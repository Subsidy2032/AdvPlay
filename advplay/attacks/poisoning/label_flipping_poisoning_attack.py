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


class LabelFlippingPoisoningAttack(PoisoningAttack, attack_type=available_attacks.POISONING, attack_subtype=poisoning_techniques.LABEL_FLIPPING):
    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

        self.log_data = {
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
            "poisoning_results": []
        }

    def execute(self):
        super().execute()
        unique_labels_original = np.unique(self.dataset[:, self.label_column])
        label_map = {label: idx for idx, label in enumerate(unique_labels_original)}
        labels = self.dataset[:, self.label_column]
        mapped_labels = np.vectorize(label_map.get)(labels)
        self.dataset[:, self.label_column] = mapped_labels

        try:
            X = np.delete(self.dataset, self.label_column, axis=1)
            y = self.dataset[:, self.label_column].astype(int)
        except Exception as e:
            raise RuntimeError(f"Failed to split features and labels: {e}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_portion,
                                                                random_state=self.seed)

        n_samples = len(y_train)
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

        for portion_to_poison in np.linspace(self.min_portion_to_poison, self.max_portion_to_poison, num_steps):
            n_to_poison = int(n_samples * portion_to_poison)

            if self.source is not None:
                source_mask = y_train == label_map[self.source]

            elif self.target is not None:
                source_mask = y_train != label_map[self.target]

            else:
                source_mask = np.ones_like(y_train, dtype=bool)

            X_source = X_train[source_mask]
            y_source = y_train[source_mask]

            if len(X_source) < n_to_poison:
                raise ValueError(f"Not enough samples to poison {portion_to_poison * 100:.1f}% of the dataset\n"
                                 f"Available samples with source class (all classes if not set): {len(X_source)}\n"
                                 f"Number of samples to poison: {n_to_poison}")


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

        most_effective_portion, min_accuracy = self.evaluate(n_samples, X_test, y_test, unique_labels_original, base_model, poisoned_models_dict)

        print(f"The attack was most effective when poisoning {most_effective_portion * 100:.1f}% of the training dataset\n")
        self.log_data["most_effective_portion"] = most_effective_portion
        self.log_data["most_effective_accuracy"] = min_accuracy

        try:
            print(f"Saving base model\n")
            save_model.save_model(self.training_framework, base_model, self.model_name)

            print(f"Saving poisoned model and poisoned dataset\n")
            save_model.save_model(self.training_framework, poisoned_models_dict[most_effective_portion], f"{self.model_name}_poisoned")
            dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.model_name}_dataset.csv"
            os.makedirs(dataset_path.parent, exist_ok=True)
            np.save(dataset_path.with_suffix(".npy"), poisoned_datasets_dict[most_effective_portion])

        except Exception as e:
            raise RuntimeError(f"Failed to save model(s) and dataset: {e}")

        append_log_entry(self.log_file_path, self.log_data)

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

        print("\n")
        base_model_accuracy = registry.evaluate_model_accuracy(self.training_framework, base_model, X_test, y_test)
        self.log_data["base_accuracy"] = base_model_accuracy
        print(f"Base model accuracy is: {base_model_accuracy:.2f}\n")

        base_model_predictions = base_model.predict(X_test)
        base_confusion_mat = confusion_matrix(y_test, base_model_predictions)
        self.log_data["labels"] = unique_labels
        self.log_data["base_confusion_matrix"] = base_confusion_mat

        for portion_to_poison, model in poisoned_models.items():
            try:
                accuracy = registry.evaluate_model_accuracy(self.training_framework, model, X_test, y_test)
                n_to_poison = int(n_samples * portion_to_poison)
                print(f"The accuracy for the model with {n_to_poison} "
                      f"samples poisoned ({portion_to_poison * 100:.1f}%) is {accuracy:.2f}")

                print(f"Attack success rate: {base_model_accuracy - accuracy:.2f}")

                predictions = model.predict(X_test)
                confusion_mat = confusion_matrix(y_test, predictions)

                self.log_data["poisoning_results"].append({
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

        return most_effective_portion, min_accuracy
