from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from advplay.attacks.base_attack import BaseAttack
from advplay.variables import available_attacks, poisoning_techniques
from advplay import paths

class PoisoningAttack(BaseAttack, ABC, attack_type=available_attacks.POISONING, attack_subtype=None):
    def __init__(self, template: dict, **kwargs):
        super().__init__(template, **kwargs)
        self.training_framework = template.get('training_framework')
        self.training_algorithm = template.get("training_algorithm")
        self.training_config = template.get("training_config")
        self.test_portion = template.get('test_portion')
        self.min_portion_to_poison = template.get('min_portion_to_poison')
        self.max_portion_to_poison = template.get('max_portion_to_poison') or self.min_portion_to_poison
        self.source_class = template.get('source_class')
        self.target_class = template.get('target_class')
        self.trigger_pattern = template.get('trigger_pattern')
        self.override = template.get('override')

        self.poisoning_method = kwargs.get('poisoning_method')
        self.dataset = kwargs.get('dataset')
        self.poisoning_data = kwargs.get('poisoning_data')
        self.seed = kwargs.get('seed')
        self.label_column = kwargs.get('label_column')
        self.step = kwargs.get('step', ((self.max_portion_to_poison - self.min_portion_to_poison) / 5))
        self.model_name = kwargs.get('model_name', datetime.now().strftime(f"{self.poisoning_method}_{self.training_algorithm}_model"))
        self.filename = kwargs.get('filename', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.validate_inputs()

        self.log_file_path = paths.ATTACK_LOGS / available_attacks.POISONING / f"{self.filename}.log"
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(self):
        pass

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