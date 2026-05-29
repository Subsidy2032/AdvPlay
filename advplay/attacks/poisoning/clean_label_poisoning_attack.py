import numpy as np
import os
from typing import Annotated
from art.attacks.poisoning import FeatureCollisionAttack
from pathlib import Path

from advplay import paths
from advplay.attacks.poisoning.poisoing_attack import PoisoningAttack
from advplay.variables import available_attacks, poisoning_techniques
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.attacks.attack_param import AttackParam
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.ml.models.loss_functions.registry import LOSS_FUNCTION_REGISTRY
from advplay.attack_evaluators.contexts.clean_label_poisoning_evaluation_context import CleanLabelPoisoningEvaluationContext

class CleanLabelPoisoningAttack(PoisoningAttack,
                                    attack_type=available_attacks.POISONING,
                                    attack_subtype=poisoning_techniques.CLEAN_LABEL):
    indices_to_poison: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=True, default=None,
                                                             help="The indices of the examples to poison. "
                                                                  "Format: '[loader:]path' (prefix overrides extension-based loader lookup).")]
    attack_configuration: Annotated[dict, AttackParam(type=dict, required=True, default={}, help="Path to a JSON config of kwargs for ART's FeatureCollisionAttack (must include feature_layer)")]

    def execute(self):
        default_path = paths.MODELS
        if not Path(self.model_path).is_file():
            self.model_path = default_path / self.model_path
        loader_cls = BaseModelLoader.registry.get(self.training_framework)
        loader = loader_cls(self.model_path, self.model, self.training_configuration)
        loss = self.model_configuration.get("loss")
        input_shape = self.model_configuration.get("input_shape")
        nb_classes = self.model_configuration.get("nb_classes")
        clip_values = self.model_configuration.get("clip_values")

        loss_fn = LOSS_FUNCTION_REGISTRY[self.training_framework](loss)
        wrapper = loader.load_art_classifier(loss_fn, input_shape, nb_classes, clip_values)

        X, y_raw = self.load_train_arrays()
        X = X.astype(np.float32)

        if isinstance(self.target, LoadedDataset):
            target = self.target.data
            target_label = None
        else:
            target_label = int(self.target)
            candidates = np.where(y_raw == target_label)[0]
            if len(candidates) == 0:
                raise ValueError(f"No samples found for target class {target_label}")

            target = X[candidates[0]]

        if target.ndim == len(input_shape):
            target = np.expand_dims(target, axis=0)

        target = target.astype(np.float32)

        attack_instance = FeatureCollisionAttack(wrapper, target, **self.attack_configuration)

        indices = np.asarray(self.indices_to_poison.data).ravel().astype(int)
        examples_to_poison = X[indices]

        poison, _ = attack_instance.poison(examples_to_poison)

        X_poisoned = X.copy()
        X_poisoned[indices] = poison

        if self.split:
            X_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.features_dataset_name}_poisoned"
            y_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.labels_dataset_name}_poisoned"
            os.makedirs(X_dataset_path.parent, exist_ok=True)

            X_loaded_dataset = LoadedDataset(X_poisoned, self.X_source_type, self.X_metadata)
            y_loaded_dataset = LoadedDataset(y_raw, self.y_source_type, self.y_metadata)
            datasets = [
                (X_loaded_dataset, str(X_dataset_path)),
                (y_loaded_dataset, str(y_dataset_path)),
            ]
        else:
            base_dataset_path = paths.DATASETS / 'poisoned_datasets' / f"{self.dataset_name}_poisoned"
            os.makedirs(base_dataset_path.parent, exist_ok=True)

            poisoned_dataset = np.insert(X_poisoned, self.label_column, y_raw, axis=1)
            loaded_dataset = LoadedDataset(poisoned_dataset, self.source_type, self.metadata)
            datasets = [(loaded_dataset, str(base_dataset_path))]

        attack_results = {
            "attack": self.attack_type,
            "technique": self.attack_subtype,
            "training_framework": self.training_framework,
            "training_configuration": self.training_configuration,
            "target": self.target,
            "attack_configuration": self.attack_configuration,
            "model_name": self.model_name,
            "n_samples_poisoned": int(len(indices)),
        }

        context = CleanLabelPoisoningEvaluationContext(
            model=self.model,
            X_test=X,
            y_test=y_raw,
            training_framework=self.training_framework,
            training_configuration=self.training_configuration,
            model_path=str(self.model_path),
            X_poisoned=X_poisoned,
            y=y_raw,
            indices_to_poison=indices,
            target_sample=target,
            target_label=target_label,
            model_name=self.model_name,
        )

        return attack_results, context, datasets
