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

    def execute(self):
        X = self.training_data.drop(columns=[self.label_column])
        y = self.training_data[self.label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_portion,
                                                            random_state=self.seed)

        n_samples = len(y_train)
        labels = y.unique()

        if len(labels) <= 1:
            raise ValueError("Only one class is present, can not perform poisoning")

        base_model = registry.train(self.training_framework, self.training_algorithm, X_train, y_train)
        base_accuracy = registry.evaluate_model_accuracy(self.training_framework, base_model, X_test, y_test)
        print(f"Base model accuracy is: {base_accuracy}")

        rng = np.random.default_rng(self.seed)

        source_mask = pd.Series([True] * n_samples, index=y_train.index)
        poisoned_models_dict = {}

        num_steps = int((self.max_portion_to_poison - self.min_portion_to_poison) / self.step) + 1
        for portion_to_poison in np.linspace(self.min_portion_to_poison, self.max_portion_to_poison, num_steps):
            n_to_poison = int(n_samples * portion_to_poison)

            if self.source_class is not None:
                source_mask = y_train == self.source_class

            X_source = X_train[source_mask]
            y_source = y_train[source_mask]

            if len(X_source) < n_to_poison:
                raise ValueError(f"Not enough samples to poison {portion_to_poison * 100:.1f}% of the dataset\n"
                                 f"Available samples with source class (all classes if not set): {len(X_source)}"
                                 f"Number of samples to poison: {n_to_poison}")


            print(f"Poisoning {n_to_poison} samples ({portion_to_poison * 100:.1f}%) from the dataset")

            source_indices = y_source.index.to_numpy()
            indices_to_flip = rng.choice(source_indices, size=n_to_poison, replace=False)

            poisoned_labels = self.poison(y_train.loc[indices_to_flip].copy(), y.unique(), rng)

            if self.override:
                X_train_poisoned = X_train.copy()
                y_poisoned = y_train.copy()
                y_poisoned.loc[indices_to_flip] = poisoned_labels

            else:
                X_train_poisoned = pd.concat([X_train, X_train.loc[indices_to_flip]])
                y_poisoned = pd.concat([y_train, poisoned_labels])

            poisoned_models_dict[portion_to_poison] = registry.train(
                self.training_framework, self.training_algorithm, X_train_poisoned, y_poisoned
            )

        min_accuracy = 1.1
        most_effective_portion = -1
        for key, value in poisoned_models_dict.items():
            accuracy = registry.evaluate_model_accuracy(self.training_framework, value, X_test, y_test)
            print(f"The accuracy for the model with {int(n_samples * key)} samples poisoned ({key * 100:.1f}%) is {accuracy}")

            if accuracy < min_accuracy:
                min_accuracy = accuracy
                most_effective_portion = key

        print(f"The attack was most effective when poisoning {most_effective_portion * 100:.1f}% of the training dataset")

        print(f"Saving base model")
        save_model.save_model(self.training_framework, base_model, self.model_name)

        print(f"Saving poisoned model")
        save_model.save_model(self.training_framework, poisoned_models_dict[most_effective_portion], f"{self.model_name}_poisoned")

    def poison(self, labels_to_poison, labels, rng):
        if self.target_class is not None:
            return pd.Series(self.target_class * len(labels_to_poison))

        for i in range(len(labels_to_poison)):
            labels_without_source = labels[labels != labels_to_poison[i]]
            labels_to_poison[i] = rng.choice(labels_without_source)

        return labels_to_poison