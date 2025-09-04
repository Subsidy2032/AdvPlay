from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os

from advplay.attacks.base_attack import BaseAttack
from advplay.variables import available_attacks, poisoning_techniques, default_template_file_names
from advplay import paths
from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.model_ops.registry import load_dataset

class PoisoningAttack(BaseAttack, ABC, attack_type=available_attacks.POISONING, attack_subtype=None):
    TEMPLATE_PARAMETERS = {
        "technique": BaseAttack.COMMON_TEMPLATE_PARAMETERS.get('technique')(available_attacks.POISONING),
        "training_framework": BaseAttack.COMMON_TEMPLATE_PARAMETERS.get('training_framework'),
        "training_algorithm": BaseAttack.COMMON_TEMPLATE_PARAMETERS.get('training_algorithm'),
        "training_configuration": BaseAttack.COMMON_TEMPLATE_PARAMETERS.get('training_configuration'),
        "test_portion": {"type": float, "required": True, "default": 0.2,
                         "help": 'Portion of the dataset to be used for testing'},
        "min_portion_to_poison": {"type": float, "required": True, "default": 0.1,
                                  "help": 'Minimum portion of the dataset to poison'},
        "max_portion_to_poison": {"type": float, "required": False, "default": None,
                                  "help": 'Maximum portion of the dataset to poison'},
        "trigger_pattern": {"required": False, "default": None, "help": "A trigger to be used for poisoning"},
        "override": {"type": bool, "required": False, "default": True,
                     "help": "Whether to override examples from the training dataset"},
        "template_filename": {"type": str, "required": False,
                              "default": default_template_file_names.CUSTOM_INSTRUCTIONS,
                              "help": "Template file name"}
    }

    ATTACK_PARAMETERS = {
        "template": BaseAttack.COMMON_ATTACK_PARAMETERS.get('template'),
        "dataset": BaseAttack.COMMON_ATTACK_PARAMETERS.get('dataset'),
        "features_dataset": BaseAttack.COMMON_ATTACK_PARAMETERS.get('features_dataset'),
        "labels_array": BaseAttack.COMMON_ATTACK_PARAMETERS.get('labels_array'),
        "label_column": BaseAttack.COMMON_ATTACK_PARAMETERS.get('label_column'),
        "source": {"type": (int, str), "required": False, "default": None, "help": 'Source class to poison'},
        "target": {"type": (int, str), "required": False, "default": None, "help": 'Target class'},
        "poisoning_dataset": {"type": pd.DataFrame, "required": False, "default": None,
                              "help": 'Poisoned samples to add to the training dataset'},
        "seed": BaseAttack.COMMON_ATTACK_PARAMETERS.get('seed'),
        "step": {"type": float, "required": False, "default": 0.1,
                 "help": 'Incrementing steps to take for poisoning portions'},
        "model_name": BaseAttack.COMMON_ATTACK_PARAMETERS.get('model_name'),
        "log_filename": BaseAttack.COMMON_ATTACK_PARAMETERS.get('log_filename')
    }

    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

        if isinstance(self.dataset, pd.DataFrame):
            if isinstance(self.label_column, str):
                self.label_column = self.dataset.columns.get_loc(self.label_column)

            self.dataset = self.dataset.to_numpy()

        elif isinstance(self.features_dataset, pd.DataFrame):
            self.features_dataset = self.features_dataset.to_numpy()

            if isinstance(self.labels_array, pd.DataFrame):
                self.labels_array = self.labels_array.to_numpy().ravel()

        elif isinstance(self.label_column, str):
            raise TypeError("str column names is only supported for dataframes at the moment")

    def execute(self):
        self.validate_attack_inputs()

    def validate_attack_inputs(self):
        if self.dataset is not None and not (isinstance(self.dataset, np.ndarray) or isinstance(self.dataset, pd.DataFrame)):
            raise TypeError("training_data must be a Pandas DataFrame or a Numpy Array")

        if not (0 < self.test_portion < 1):
            raise ValueError("test_portion must be between 0 and 1")

        if not (0 <= self.min_portion_to_poison <= 1):
            raise ValueError("min_portion_to_poison must be between 0 and 1")

        if not (0 <= self.max_portion_to_poison <= 1):
            raise ValueError("max_portion_to_poison must be between 0 and 1")

        if self.min_portion_to_poison > self.max_portion_to_poison:
            raise ValueError("min_portion_to_poison cannot be greater than max_portion_to_poison")

        if self.step < 0:
            raise ValueError("step must be a positive number")

        if self.seed is not None and not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer or None")

    def validate_template_inputs(self):
        if self.max_portion_to_poison == self.min_portion_to_poison and self.step > 0:
            raise ValueError("Can't use the step parameter if max_portion_to_poison is equal to min_portion_to_poison or not set")

        if (self.training_framework, self.training_algorithm) not in BaseTrainer.registry.keys():
            raise ValueError(
                f"Invalid framework and training algorithm configuration: ({self.training_framework}, {self.training_algorithm})")

        for name, val in [
            ("test_portion", self.test_portion),
            ("min_portion_to_poison", self.min_portion_to_poison),
            ("max_portion_to_poison", self.max_portion_to_poison)
        ]:
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(val).__name__}")
            if not (0 <= val <= 1):
                raise ValueError(f"{name} must be between 0 and 1, got {val}")

        if (self.min_portion_to_poison is not None
                and self.max_portion_to_poison is not None
                and self.min_portion_to_poison > self.max_portion_to_poison):
            raise ValueError("min_portion_to_poison cannot be greater than max_portion_to_poison")

        if self.source is not None and self.target is not None:
            if self.source == self.target:
                raise ValueError("source and target_class must be different")

        if self.override is not None and not isinstance(self.override, bool):
            raise TypeError(f"override must be a boolean, got {type(self.override).__name__}")

        if not isinstance(self.template_filename, str) or not self.template_filename.strip():
            raise ValueError("Tempalte filename must be a non-empty string")
        if any(c in self.template_filename for c in r'\/:*?"<>|'):
            raise ValueError(f"Template filename contains invalid characters: {self.template_filename}")

        if self.training_configuration is not None and not isinstance(self.training_configuration, dict):
            raise FileNotFoundError(f"training_configuration must be a dictionary, got {type(self.training_configuration)}")