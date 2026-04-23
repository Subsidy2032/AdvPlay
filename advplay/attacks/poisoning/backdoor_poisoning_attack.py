import numpy as np
import os
from typing import Annotated
from art.attacks.poisoning import FeatureCollisionAttack
from pathlib import Path
from sklearn.model_selection import train_test_split
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_single_bd, add_pattern_bd, insert_image

from advplay import paths
from advplay.attacks.poisoning.poisoing_attack import PoisoningAttack
from advplay.variables import available_attacks, poisoning_techniques
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.attacks.attack_param import AttackParam
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.ml.models.loss_functions.registry import LOSS_FUNCTION_REGISTRY
from advplay.attack_evaluators.contexts.backdoor_poisoning_evaluation_context import BackdoorPoisoningEvaluationContext

class BackdoorPoisoningAttack(PoisoningAttack,
                                    attack_type=available_attacks.POISONING,
                                    attack_subtype=poisoning_techniques.BACKDOOR):
    trigger: Annotated[str, AttackParam(type=str, required=True, default='single_pixel',
                                         help="Trigger type: 'single_pixel', 'pattern', or 'image'")]
    trigger_image: Annotated[str, AttackParam(type=str, required=False, default=None,
                                               help="Path to a trigger image (used when trigger='image'). Relative paths resolve under resources/triggers/")]
    trigger_blend: Annotated[float, AttackParam(type=float, required=False, default=0.8,
                                                 help="Blend factor for the image trigger (0-1)")]
    trigger_random: Annotated[bool, AttackParam(type=bool, required=False, default=True,
                                                 help="Place the image trigger at a random position (if false, use trigger_x_shift/trigger_y_shift)")]
    trigger_x_shift: Annotated[int, AttackParam(type=int, required=False, default=0,
                                                 help="Pixels from the left to place the image trigger (trigger_random=false)")]
    trigger_y_shift: Annotated[int, AttackParam(type=int, required=False, default=0,
                                                 help="Pixels from the top to place the image trigger (trigger_random=false)")]

    _TRIGGER_ALIASES = {
        'single': 'single_pixel', 'single_pixel': 'single_pixel',
        'pattern': 'pattern', 'checkerboard': 'pattern',
        'image': 'image',
    }

    def execute(self):
        super().execute()

        if self.target is None:
            raise ValueError("Backdoor attack requires a target class")

        trigger_kind = self._TRIGGER_ALIASES.get(self.trigger)
        if trigger_kind is None:
            raise ValueError(
                f"Unknown trigger '{self.trigger}'. Expected one of: "
                f"{sorted(set(self._TRIGGER_ALIASES.values()))}"
            )
        if trigger_kind == 'image' and not self.trigger_image:
            raise ValueError("trigger='image' requires --trigger-image <path>")

        X_train_raw, y_train_raw = self.load_train_arrays()
        X_test_raw, y_test_raw = self.load_test_arrays()
        combined_labels = (np.concatenate([y_train_raw, y_test_raw])
                           if y_test_raw is not None else y_train_raw)
        labels_unique = np.unique(combined_labels)
        label_map = {lbl: i for i, lbl in enumerate(labels_unique)}
        reverse_label_map = {i: lbl for lbl, i in label_map.items()}

        if len(labels_unique) <= 1:
            raise ValueError("Poisoning requires at least two classes")

        y_train_mapped = np.vectorize(label_map.get)(y_train_raw).astype(int)
        if self.pre_split:
            X_train, y_train = X_train_raw, y_train_mapped
            X_test = X_test_raw
            y_test = np.vectorize(label_map.get)(y_test_raw).astype(int)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train_raw, y_train_mapped, test_size=self.test_portion, random_state=self.seed
            )

        source_mask = (
            (y_train == label_map[self.source]) if self.source is not None else
            (y_train != label_map[self.target])
        )
        y_source = y_train[source_mask]
        n_samples = len(y_source)

        poisoned_datasets = {}
        steps = max(1, int(round((self.max_portion_to_poison - self.min_portion_to_poison) / self.step)) + 1)
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
            X_poisoned = X_train.copy() if self.override else np.vstack([X_train, X_train[idx]])

            poisoned_slots = idx if self.override else np.arange(len(y_train), len(y_train) + len(idx))
            target_labels = np.full(len(idx), label_map[self.target], dtype=y_train.dtype)

            X_poisoned[poisoned_slots], y_poisoned[poisoned_slots] = self.poison_examples(X_train[idx], target_labels)

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

        X_test_triggered = self._perturb(X_test.copy())

        target_label = label_map[self.target]
        source_label = label_map[self.source] if self.source is not None else None

        example_source_mask = (
            (y_test == source_label) if source_label is not None else (y_test != target_label)
        )
        example_candidates = np.where(example_source_mask)[0]
        if len(example_candidates) == 0:
            example_idx = 0
        else:
            example_idx = int(example_candidates[0])
        example_clean = X_test[example_idx]
        example_triggered = X_test_triggered[example_idx]
        example_true_label = reverse_label_map[int(y_test[example_idx])]

        poisoning_context = BackdoorPoisoningEvaluationContext(
            model=self.model,
            X_test=X_test,
            y_test=y_test,
            training_framework=self.training_framework,
            training_configuration=self.training_configuration,
            clean_dataset=clean_dataset,
            poisoned_datasets=poisoned_datasets,
            model_name=self.model_name,
            X_test_triggered=X_test_triggered,
            source_label=source_label,
            target_label=target_label,
            source_class=self.source,
            target_class=self.target,
            labels=labels_unique,
            trigger=self._TRIGGER_ALIASES[self.trigger],
            example_clean=example_clean,
            example_triggered=example_triggered,
            example_true_label=example_true_label,
        )

        attack_results = {
            "attack": self.attack_type, "technique": self.attack_subtype,
            "training_framework": self.training_framework,
            "training_configuration": self.training_configuration,
            "test_portion": self.test_portion, "min_portion_to_poison": self.min_portion_to_poison,
            "max_portion_to_poison": self.max_portion_to_poison, "source": self.source, "target": self.target,
            "override": self.override, "seed": self.seed, "step": self.step, "model_name": self.model_name,
            "trigger": self._TRIGGER_ALIASES[self.trigger],
            "labels": labels_unique
        }

        return attack_results, poisoning_context, datasets

    def _pixel_value(self, dtype):
        return 255 if np.issubdtype(dtype, np.integer) else 1.0

    def _resolve_trigger_image(self):
        p = Path(self.trigger_image)
        if p.is_file():
            return str(p)
        fallback = paths.RESOURCES / 'triggers' / self.trigger_image
        if fallback.is_file():
            return str(fallback)
        raise FileNotFoundError(f"Trigger image not found: {self.trigger_image}")

    def _perturb(self, x):
        channels_first = x.ndim == 4 and x.shape[1] in (1, 3)
        trigger_kind = self._TRIGGER_ALIASES[self.trigger]
        pixel_value = self._pixel_value(x.dtype)

        if trigger_kind == 'single_pixel':
            if channels_first:
                x_cl = np.transpose(x, (0, 2, 3, 1))
                poisoned = add_single_bd(x_cl, distance=2, pixel_value=pixel_value)
                return np.transpose(poisoned, (0, 3, 1, 2))
            return add_single_bd(x, distance=2, pixel_value=pixel_value)

        if trigger_kind == 'pattern':
            return add_pattern_bd(x, distance=2, pixel_value=pixel_value, channels_first=channels_first)

        backdoor_path = self._resolve_trigger_image()
        kwargs = dict(
            backdoor_path=backdoor_path,
            channels_first=channels_first,
            random=self.trigger_random,
            x_shift=self.trigger_x_shift,
            y_shift=self.trigger_y_shift,
            blend=self.trigger_blend,
        )
        if np.issubdtype(x.dtype, np.integer):
            x_f = x.astype(np.float32) / 255.0
            poisoned = insert_image(x_f, **kwargs)
            return np.clip(poisoned * 255.0, 0, 255).astype(x.dtype)
        return insert_image(x, **kwargs)

    def poison_examples(self, examples, target):
        backdoor = PoisoningAttackBackdoor(perturbation=self._perturb)
        return backdoor.poison(examples, target)
