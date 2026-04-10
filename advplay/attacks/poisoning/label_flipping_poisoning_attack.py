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

        if self.split:
            X_base_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.features_dataset_name}_poisoned"
            y_base_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.labels_dataset_name}_poisoned"
            os.makedirs(X_base_dataset_path.parent, exist_ok=True)
            os.makedirs(y_base_dataset_path.parent, exist_ok=True)

        else:
            base_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.dataset_name}_poisoned"
            os.makedirs(base_dataset_path.parent, exist_ok=True)

        datasets = []
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

            y_poisoned_original = np.vectorize(reverse_label_map.get)(y_poisoned)
            if self.split:
                X_loaded_dataset = LoadedDataset(
                    X_poisoned,
                    self.X_source_type,
                    self.X_metadata
                )
                X_dataset_path = str(X_base_dataset_path) + f"_{portion}"
                datasets.append((X_loaded_dataset, X_dataset_path))

                y_loaded_dataset = LoadedDataset(
                    y_poisoned_original,
                    self.y_source_type,
                    self.y_metadata
                )
                y_dataset_path = str(y_base_dataset_path) + f"_{portion}"
                datasets.append((y_loaded_dataset, y_dataset_path))

            else:
                poisoned_dataset = np.insert(X_poisoned, self.label_column, y_poisoned_original, axis=1)
                loaded_dataset = LoadedDataset(poisoned_dataset, self.source_type, self.metadata)
                dataset_path = str(base_dataset_path) + f"_{portion}"
                datasets.append((loaded_dataset, dataset_path))

        clean_dataset = {
            "X_train": X_train,
            "y_train": y_train
        }
        poisoning_context = PoisoningContext(self.model, X_test, y_test, self.training_framework, self.training_configuration, clean_dataset, poisoned_datasets, self.model_name)
         
        attack_results = {
            "attack": self.attack_type, "technique": self.attack_subtype,
            "training_framework": self.training_framework,
            "training_configuration": self.training_configuration,
            "test_portion": self.test_portion, "min_portion_to_poison": self.min_portion_to_poison,
            "max_portion_to_poison": self.max_portion_to_poison, "source": self.source, "target": self.target,
            "override": self.override, "seed": self.seed, "step": self.step, "model_name": self.model_name,
            "labels": labels_unique
        }

        return attack_results, poisoning_context, datasets

    def poison(self, labels_to_poison, labels, label_map, rng):
        if self.target is not None:
            return np.full_like(labels_to_poison, fill_value=label_map[self.target])
        poisoned = labels_to_poison.copy()
        for i in range(len(poisoned)):
            poisoned[i] = rng.choice(labels[labels != poisoned[i]])
        return poisoned
