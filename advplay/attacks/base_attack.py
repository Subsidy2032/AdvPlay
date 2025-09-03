from abc import ABC, abstractmethod
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd

from advplay import paths
from advplay.model_ops.trainers.base_trainer import BaseTrainer

class BaseAttack(ABC):
    registry = {}
    techniques_per_attack = {}

    COMMON_TEMPLATE_PARAMETERS = {
        "technique": lambda attack: {"type": str, "required": True, "default": None,
                                     "help": "The poisoning technique",
                                     "choices": lambda: BaseAttack.techniques_per_attack.get(attack, [])},
        "training_framework": {"type": str, "required": True, "default": "sklearn",
                               "help": 'Framework for training the model',
                               "choices": lambda: list(
                                   {k[0] for k in BaseTrainer.registry.keys() if k[0] is not None})},
        "training_algorithm": {"type": str, "required": True, "default": "logistic_regression",
                               "help": 'The training algorithm',
                               "choices": lambda: list(
                                   {k[1] for k in BaseTrainer.registry.keys() if k[1] is not None})},
        "training_configuration": {"type": dict, "required": False, "default": None,
                                   "help": 'Path to a training configuration file'}
    }

    COMMON_ATTACK_PARAMETERS = {
        "template": {"type": str, "required": True, "default": None, "help": "The name of the template for the attack"},
        "dataset": {"type": (np.array, pd.DataFrame), "required": True, "default": None, "help": 'Dataset to poison'},
        "label_column": {"type": (int, str), "required": True, "default": None, "help": 'The name of the label column'},
        "seed": {"type": int, "required": False, "default": None, "help": 'Seed for reproduction'},
        "model_name": {"type": str, "required": False, "default": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                       "help": 'The name of the model that will be saved'},
        "log_filename": {"type": str, "required": False, "default": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                         "help": "Log file name to save attack results to"}

    }

    def __init_subclass__(cls, attack_type: str, attack_subtype, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.attack_type = attack_type
        cls.attack_subtype = attack_subtype

        key = (attack_type, attack_subtype)
        BaseAttack.registry[key] = cls

        if attack_type not in BaseAttack.techniques_per_attack:
            BaseAttack.techniques_per_attack[attack_type] = []

        if attack_subtype is not None and attack_subtype not in BaseAttack.techniques_per_attack[attack_type]:
            BaseAttack.techniques_per_attack[attack_type].append(attack_subtype)

    def __init__(self, template: dict, **kwargs):
        template_params = getattr(super(self.__class__, self), "TEMPLATE_PARAMETERS", {})
        attack_params = getattr(super(self.__class__, self), "ATTACK_PARAMETERS", {})

        for key, meta in template_params.items():
            value = template.get(key)
            if value is None:
                value = meta.get("default")
            setattr(self, key, value)

        for key, meta in attack_params.items():
            value = kwargs.get(key)
            if value is None:
                value = meta.get("default")
            setattr(self, key, value)


        self.log_file_path = None
        self.setup_logging()

        self.validate_template_inputs()

    def setup_logging(self):
        filename = getattr(self, "log_filename", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_file_path = paths.ATTACK_LOGS / f"{self.attack_type}" / f"{filename}.log"
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self):
        template_values = {key: getattr(self, key)
                           for key in getattr(self.__class__, "TEMPLATE_PARAMETERS", {})
                           if key != "template_filename"}
        self.save_template(self.template_filename, template_values)

    def save_template(self, filename: str, template: dict):
        template_json = json.dumps(template, indent=4)
        filename = paths.TEMPLATES / self.attack_type / f"{filename}.json"

        if filename.exists():
            while True:
                override = input("Configuration file already exists, are you sure you want to replace it? (Y/N) ")
                if override.lower() == 'n':
                    return
                elif override.lower() == 'y':
                    break
                else:
                    print("Invalid option, please choose from the provided options.\n")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(template_json)

    def validate_attack_inputs(self):
        pass

    def validate_template_inputs(self):
        pass

    @abstractmethod
    def execute(self):
        raise NotImplementedError("Subclasses must implement the execute method.")
