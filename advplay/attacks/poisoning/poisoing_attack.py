from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from advplay.attacks.base_attack import BaseAttack
from advplay.variables import available_attacks, poisoning_techniques, default_template_file_names
from advplay import paths
from advplay.model_ops.trainers.base_trainer import BaseTrainer

class PoisoningAttack(BaseAttack, ABC, attack_type=available_attacks.POISONING, attack_subtype=None):
    TEMPLATE_PARAMETERS = {
        "technique": {"type": str, "required": True, "default": poisoning_techniques.LABEL_FLIPPING,
                             "help": "The poisoning technique"},
        "training_framework": {"type": str, "required": True, "default": "sklearn",
                               "help": 'Framework for training the model'},
        "training_algorithm": {"type": str, "required": True, "default": "logistic_regression",
                               "help": 'The training algorithm'},
        "training_configuration": {"type": dict, "required": False, "default": None,
                                   "help": 'Path to a training configuration file'},
        "test_portion": {"type": float, "required": True, "default": 0.2,
                         "help": 'Portion of the dataset to be used for testing'},
        "min_portion_to_poison": {"type": float, "required": True, "default": 0.1,
                                  "help": 'Minimum portion of the dataset to poison'},
        "max_portion_to_poison": {"type": float, "required": False, "default": None,
                                  "help": 'Maximum portion of the dataset to poison'},
        "source": {"type": (str, int), "required": False, "default": None, "help": 'Source class to poison'},
        "target": {"type": (str, int), "required": False, "default": None, "help": 'Target class'},
        "trigger_pattern": {"required": False, "default": None, "help": "A trigger to be used for poisoning"},
        "override": {"type": bool, "required": False, "default": None,
                     "help": "Whether to override examples from the training dataset"},
        "template_filename": {"type": str, "required": False,
                              "default": default_template_file_names.CUSTOM_INSTRUCTIONS,
                              "help": "Template file name"}
    }

    ATTACK_PARAMETERS = {
        "template": {"type": str, "required": True, "default": None, "help": "The name of the template for the attack"},
        "dataset": {"type": pd.DataFrame, "required": True, "default": None, "help": 'Dataset to poison'},
        "label_column": {"type": str, "required": True, "default": None, "help": 'The name of the label column'},
        "poisoning_dataset": {"type": pd.DataFrame, "required": False, "default": None,
                              "help": 'Poisoned samples to add to the training dataset'},
        "seed": {"type": int, "required": False, "default": None, "help": 'Seed for reproduction'},
        "step": {"type": float, "required": False, "default": 0.1,
                 "help": 'Incrementing steps to take for poisoning portions'},
        "model_name": {"type": str, "required": False, "default": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                       "help": 'The name of the model that will be saved'},
        "log_filename": {"type": str, "required": False, "default": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                         "help": "Log file name to save attack results to"}
    }

    def validate_inputs(self):
        if self.dataset is None or not isinstance(self.dataset, pd.DataFrame):
            raise TypeError("training_data must be a pandas DataFrame")

        if self.label_column not in self.dataset.columns:
            raise ValueError(f"label_column '{self.label_column}' not found in training_data")

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
        if any(c in self.filename for c in r'\/:*?"<>|'):
            raise ValueError(f"Template filename contains invalid characters: {self.template_filename}")

        if self.training_configuration is not None and not Path(self.training_configuration).exists():
            raise FileNotFoundError(f"training_config file does not exist: {self.training_configuration}")