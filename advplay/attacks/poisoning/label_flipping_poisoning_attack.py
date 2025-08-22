import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from advplay.model_ops import registry
from advplay.utils import save_model
from advplay import paths
from advplay.utils.append_log_entry import append_log_entry

class LabelFlippingPoisoningAttack():
    def __init__(self, training_framework, training_algorithm, training_config, test_portion, min_portion_to_poison,
                 max_portion_to_poison, source_class, target_class, trigger_pattern, override,
                 dataset, poisoning_data, seed, label_column, step, model_name, log_file_path):
        self.training_framework = training_framework
        self.training_algorithm = training_algorithm
        self.training_config = training_config
        self.test_portion = test_portion
        self.min_portion_to_poison = min_portion_to_poison
        self.max_portion_to_poison = max_portion_to_poison
        self.source_class = source_class
        self.target_class = target_class
        self.trigger_pattern = trigger_pattern
        self.override = override
        self.dataset = dataset
        self.poisoning_data = poisoning_data
        self.seed = seed
        self.label_column = label_column
        self.step = step
        self.model_name = model_name
        self.log_file_path = log_file_path

        self.log_data = {
            "training_framework": training_framework,
            "training_algorithm": training_algorithm,
            "training_config": training_config,
            "test_portion": test_portion,
            "min_portion_to_poison": min_portion_to_poison,
            "max_portion_to_poison": max_portion_to_poison,
            "source_class": source_class,
            "target_class": target_class,
            "override": override,
            "seed": seed,
            "step": step,
            "model_name": model_name,
            "poisoning_results": []
        }

    def execute(self):
        try:
            X = self.dataset.drop(columns=[self.label_column])
            y = pd.Series(self.dataset[self.label_column].values.astype(int))
        except Exception as e:
            raise RuntimeError(f"Failed to split features and labels: {e}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_portion,
                                                                random_state=self.seed)

        n_samples = len(y_train)
        labels = y.unique()

        if len(labels) <= 1:
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

            if self.source_class is not None:
                source_mask = y_train == self.source_class

            elif self.target_class is not None:
                source_mask = y_train != self.target_class

            else:
                source_mask = pd.Series(True, index=y_train.index)

            X_source = X_train[source_mask]
            y_source = y_train[source_mask]

            if len(X_source) < n_to_poison:
                raise ValueError(f"Not enough samples to poison {portion_to_poison * 100:.1f}% of the dataset\n"
                                 f"Available samples with source class (all classes if not set): {len(X_source)}"
                                 f"Number of samples to poison: {n_to_poison}")


            print(f"Poisoning {n_to_poison} samples ({portion_to_poison * 100:.1f}%) from the dataset")

            try:
                source_indices = y_source.index.to_numpy()
                indices_to_flip = rng.choice(source_indices, size=n_to_poison, replace=False)
                poisoned_labels = self.poison(y_train.loc[indices_to_flip].copy(), y.unique(), rng)
                poisoned_labels = poisoned_labels.astype(int)
            except Exception as e:
                raise RuntimeError(f"Failed during poisoning step at {portion_to_poison * 100:.1f}%: {e}")

            if self.override:
                X_train_poisoned = X_train.copy()
                y_train_poisoned = y_train.copy()
                y_train_poisoned.loc[indices_to_flip] = poisoned_labels

            else:
                X_train_poisoned = pd.concat([X_train, X_train.loc[indices_to_flip]])
                y_train_poisoned = pd.concat([y_train, poisoned_labels])

            y_train_poisoned = y_train_poisoned.astype(int)

            try:
                model = registry.train(self.training_framework, self.training_algorithm, X_train_poisoned, y_train_poisoned)
            except Exception as e:
                raise RuntimeError(f"Failed to train poisoned model at {portion_to_poison * 100:.1f}%: {e}")

            poisoned_models_dict[portion_to_poison] = model
            poisoned_datasets_dict[portion_to_poison] = X_train_poisoned.copy()
            poisoned_datasets_dict[portion_to_poison][self.label_column] = y_train_poisoned

        if not poisoned_models_dict:
            raise RuntimeError("No poisoned models were generated; check your configuration")

        most_effective_portion, min_accuracy = self.evaluate(n_samples, X_test, y_test, base_model, poisoned_models_dict)

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
            poisoned_datasets_dict[most_effective_portion].to_csv(dataset_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save model(s) and dataset: {e}")

        append_log_entry(self.log_file_path, self.log_data)

    def poison(self, labels_to_poison, labels, rng):
        if self.target_class is not None:
            return pd.Series([self.target_class] * len(labels_to_poison), index=labels_to_poison.index)

        poisoned = labels_to_poison.copy()
        for idx in poisoned.index:
            labels_without_source = labels[labels != poisoned.loc[idx]]
            poisoned.loc[idx] = rng.choice(labels_without_source)

        return poisoned

    def evaluate(self, n_samples, X_test, y_test, base_model, poisoned_models: dict):
        min_accuracy = 1.1
        most_effective_portion = None

        if self.source_class is not None:
            source_mask = y_test == self.source_class
            X_test = X_test.loc[source_mask]
            y_test = y_test.loc[source_mask]

        if self.target_class is not None:
            non_target_mask = y_test != self.target_class
            X_test = X_test.loc[non_target_mask]
            y_test = y_test.loc[non_target_mask]

        print("\n")
        base_model_accuracy = registry.evaluate_model_accuracy(self.training_framework, base_model, X_test, y_test)
        self.log_data["base_accuracy"] = base_model_accuracy
        print(f"Base model accuracy on sample from the source class "
              f"{self.source_class if self.source_class is not None else 'all classes'} is: {base_model_accuracy}\n")

        for portion_to_poison, model in poisoned_models.items():
            try:
                predictions = model.predict(X_test)
                accuracy = registry.evaluate_model_accuracy(self.training_framework, model, X_test, y_test)
                n_to_poison = int(n_samples * portion_to_poison)
                print(f"The accuracy for the model on samples from the source class "
                      f"{self.source_class if self.source_class is not None else 'all classes'} with {n_to_poison} "
                      f"samples poisoned ({portion_to_poison * 100:.1f}%) is {accuracy}")

                print(f"Attack success rate: {base_model_accuracy - accuracy}")

                if self.target_class is not None:
                    num_target = sum(pred == self.target_class for pred in predictions)
                    print(f"Portion of samples not from the target class {self.target_class} classified as "
                          f"{self.target_class}: {num_target / len(y_test)}\n")

                self.log_data["poisoning_results"].append({
                    "portion_to_poison": portion_to_poison,
                    "n_samples_poisoned": n_to_poison,
                    "accuracy": accuracy
                })

                if accuracy < min_accuracy:
                    min_accuracy = accuracy
                    most_effective_portion = portion_to_poison
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate poisoned model at {portion_to_poison * 100:.1f}%: {e}")

        return most_effective_portion, min_accuracy
