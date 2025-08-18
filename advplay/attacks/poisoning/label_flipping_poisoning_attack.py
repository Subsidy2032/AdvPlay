import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from advplay.model_ops import registry
from advplay.utils import save_model

class LabelFlippingPoisoningAttack():
    def __init__(self, training_framework, training_algorithm, training_config, test_portion, min_portion_to_poison,
                 max_portion_to_poison, source_class, target_class, trigger_pattern, override,
                 training_data, poisoning_data, seed, label_column, step, model_name, log_file_path):
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
        self.training_data = training_data
        self.poisoning_data = poisoning_data
        self.seed = seed
        self.label_column = label_column
        self.step = step
        self.model_name = model_name
        self.log_file_path = log_file_path

        self.validate_inputs()

    def execute(self):
        try:
            X = self.training_data.drop(columns=[self.label_column])
            y = pd.Series(self.training_data[self.label_column].values.astype(int))
        except Exception as e:
            raise RuntimeError(f"Failed to split features and labels: {e}")

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_portion,
                                                                random_state=self.seed)
        except Exception as e:
            raise RuntimeError(f"Train/test split failed: {e}")

        n_samples = len(y_train)
        labels = y.unique()

        if len(labels) <= 1:
            raise ValueError("Only one class is present; poisoning requires at least two distinct classes")

        try:
            base_model = registry.train(self.training_framework, self.training_algorithm, X_train, y_train)
            base_accuracy = registry.evaluate_model_accuracy(self.training_framework, base_model, X_test, y_test)
            print(f"Base model accuracy is: {base_accuracy}")
        except Exception as e:
            raise RuntimeError(f"Failed to train or evaluate base model: {e}")

        rng = np.random.default_rng(self.seed)
        poisoned_models_dict = {}

        num_steps = int((self.max_portion_to_poison - self.min_portion_to_poison) / self.step) + 1
        if num_steps <= 0:
            raise ValueError("Number of poisoning steps must be positive; check min/max_portion and step values")

        for portion_to_poison in np.linspace(self.min_portion_to_poison, self.max_portion_to_poison, num_steps):
            n_to_poison = int(n_samples * portion_to_poison)

            source_mask = y_train == self.source_class if self.source_class is not None else pd.Series(True,
                                                                                                       index=y_train.index)
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
                y_poisoned = y_train.copy()
                y_poisoned.loc[indices_to_flip] = poisoned_labels

            else:
                X_train_poisoned = pd.concat([X_train, X_train.loc[indices_to_flip]])
                y_poisoned = pd.concat([y_train, poisoned_labels])

            y_poisoned = y_poisoned.astype(int)

            try:
                poisoned_models_dict[portion_to_poison] = registry.train(
                    self.training_framework, self.training_algorithm, X_train_poisoned, y_poisoned
                )
            except Exception as e:
                raise RuntimeError(f"Failed to train poisoned model at {portion_to_poison * 100:.1f}%: {e}")

            if not poisoned_models_dict:
                raise RuntimeError("No poisoned models were generated; check your configuration")

        min_accuracy = 1.1
        most_effective_portion = None

        for key, value in poisoned_models_dict.items():
            try:
                accuracy = registry.evaluate_model_accuracy(self.training_framework, value, X_test, y_test)
                print(f"The accuracy for the model with {int(n_samples * key)} samples poisoned ({key * 100:.1f}%) is {accuracy}")

                if accuracy < min_accuracy:
                    min_accuracy = accuracy
                    most_effective_portion = key
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate poisoned model at {key * 100:.1f}%: {e}")

        print(f"The attack was most effective when poisoning {most_effective_portion * 100:.1f}% of the training dataset")

        try:
            print(f"Saving base model")
            save_model.save_model(self.training_framework, base_model, self.model_name)

            print(f"Saving poisoned model")
            save_model.save_model(self.training_framework, poisoned_models_dict[most_effective_portion], f"{self.model_name}_poisoned")
        except Exception as e:
            raise RuntimeError(f"Failed to save model(s): {e}")

    def poison(self, labels_to_poison, labels, rng):
        if self.target_class is not None:
            return pd.Series([self.target_class] * len(labels_to_poison), index=labels_to_poison.index)

        for i in range(len(labels_to_poison)):
            labels_without_source = labels[labels != labels_to_poison[i]]
            labels_to_poison.at[i] = rng.choice(labels_without_source)

        return labels_to_poison

    def validate_inputs(self):
        if self.training_data is None or not isinstance(self.training_data, pd.DataFrame):
            raise TypeError("training_data must be a pandas DataFrame")

        if self.label_column not in self.training_data.columns:
            raise ValueError(f"label_column '{self.label_column}' not found in training_data")

        if not (0 < self.test_portion < 1):
            raise ValueError("test_portion must be between 0 and 1")

        if not (0 <= self.min_portion_to_poison <= 1):
            raise ValueError("min_portion_to_poison must be between 0 and 1")

        if not (0 <= self.max_portion_to_poison <= 1):
            raise ValueError("max_portion_to_poison must be between 0 and 1")

        if self.min_portion_to_poison > self.max_portion_to_poison:
            raise ValueError("min_portion_to_poison cannot be greater than max_portion_to_poison")

        if self.step <= 0:
            raise ValueError("step must be a positive number")

        if self.seed is not None and not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer or None")