from datetime import datetime
from pathlib import Path
import pandas as pd

from advplay.attacks.base_attack import BaseAttack
from advplay.variables import available_attacks, poisoning_techniques
from advplay.attacks.poisoning.label_flipping_poisoning_attack import LabelFlippingPoisoningAttack
from advplay.paths import ATTACK_LOGS

class PoisoningAttack(BaseAttack, attack_type=available_attacks.POISONING):
    def __init__(self, template: dict, **kwargs):
        super().__init__(template, **kwargs)
        self.poisoning_method = template.get('poisoning_method')
        self.training_framework = template.get('training_framework')
        self.training_algorithm = template.get("training_algorithm")
        self.training_config = template.get("training_config")
        self.test_portion = template.get('test_portion')
        self.min_portion_to_poison = template.get('min_portion_to_poison')
        self.max_portion_to_poison = template.get('max_portion_to_poison', self.min_portion_to_poison)
        self.source_class = template.get('source_class')
        self.target_class = template.get('target_class')
        self.trigger_pattern = template.get('trigger_pattern')
        self.override = template.get("override")

        self.dataset = kwargs.get('dataset')
        self.poisoning_data = kwargs.get('poisoning_data')
        self.seed = kwargs.get('seed')
        self.label_column = kwargs.get('label_column')
        self.step = kwargs.get('step', ((self.max_portion_to_poison - self.min_portion_to_poison) / 5))
        self.model_name = kwargs.get('model_name', datetime.now().strftime(f"{self.poisoning_method}_{self.training_algorithm}_model"))
        self.filename = kwargs.get('filename', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.poisoning_techniques_cls = {
            poisoning_techniques.LABEL_FLIPPING: LabelFlippingPoisoningAttack
        }

    def execute(self):
        poisoning_method_cls = self.poisoning_techniques_cls.get(self.poisoning_method)

        if poisoning_method_cls is None:
            raise ValueError(f"Unsupported poisoning method: {self.poisoning_method}")

        log_file_path = ATTACK_LOGS / available_attacks.POISONING / f"{self.filename}.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)


        executor = poisoning_method_cls(self.training_framework, self.training_algorithm, self.training_config,
                                        self.test_portion, self.min_portion_to_poison, self.max_portion_to_poison,
                                        self.source_class, self.target_class, self.trigger_pattern, self.override,
                                        self.dataset, self.poisoning_data, self.seed, self.label_column,
                                        self.step, self.model_name, log_file_path)
        executor.execute()
