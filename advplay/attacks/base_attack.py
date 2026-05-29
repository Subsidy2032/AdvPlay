from abc import ABC, abstractmethod
from datetime import datetime
from typing import get_args, get_type_hints
import json
import os
import numpy as np

from advplay import paths
from advplay.attacks.attack_param import AttackParam, TemplateParam
from advplay.ml.ops.trainers.base_trainer import BaseTrainer
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.ml.data.preprocessors.base_preprocessor import BasePreprocessor
from advplay.ml.models.architecture.registry import MODEL_REGISTRY

class BaseAttack(ABC):
    registry = {}
    techniques_per_attack = {}

    COMMON_TEMPLATE_PARAMETERS = {
        "technique": lambda attack: TemplateParam(type=str, required=True, default=None,
                                                  help="The poisoning technique",
                                                  choices=lambda: BaseAttack.techniques_per_attack.get(attack, [])),
        "training_framework": TemplateParam(type=str, required=True, default="sklearn",
                                            help='Framework for training the model',
                                            choices=lambda: list(
                                                {k[0] for k in BaseTrainer.registry.keys() if k[0] is not None})),
        "model": TemplateParam(type=str, required=True, default="logistic_regression",
                               help='The training algorithm',
                               choices=lambda: list(
                                   {k[1] for k in BaseTrainer.registry.keys() if k[1] is not None}) + list({k for k in MODEL_REGISTRY.keys() if k is not None})),
        "training_configuration": TemplateParam(type=dict, required=False, default=None,
                                                help='Path to a training configuration file'),
        "preprocessing": TemplateParam(type=dict, required=False, default=None,
                                       help='Path to a JSON file describing preprocessing steps applied to input '
                                            'datasets before the attack runs. The JSON top-level may be a list '
                                            '(ordered chain; each entry is a preprocessor name or '
                                            '{"name": <name>, "params": {...}}) or a dict mapping '
                                            'preprocessor name to its params (order preserved).')
    }

    COMMON_ATTACK_PARAMETERS = {
        "template": AttackParam(type=str, required=True, default=None,
                                help="The name of the template for the attack"),
        "dataset": AttackParam(type=LoadedDataset, required=False, default=None,
                                help="Dataset to poison. Format: '[loader:]path' "
                                     "(prefix overrides extension-based loader lookup)."),
        "features_dataset": AttackParam(type=LoadedDataset, required=False, default=None,
                                        help="Examples dataset. Format: '[loader:]path' "
                                             "(prefix overrides extension-based loader lookup)."),
        "labels_array": AttackParam(type=LoadedDataset, required=False, default=None,
                                    help="Dataset to poison. Format: '[loader:]path' "
                                         "(prefix overrides extension-based loader lookup)."),
        "label_column": AttackParam(type=(int, str), required=False, default=None,
                                    help='The name of the label column'),
        "seed": AttackParam(type=int, required=False, default=None, help='Seed for reproduction'),
        "model_name": AttackParam(type=str, required=False,
                                  default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                  help='The name of the model that will be saved'),
        "log_filename": AttackParam(type=str, required=False,
                                    default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                    help="Log file name to save attack results to")
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

        template_params = {}
        attack_params = {}
        for name, annotation in get_type_hints(cls, include_extras=True).items():
            for meta in get_args(annotation):
                if isinstance(meta, TemplateParam):
                    template_params[name] = meta
                elif isinstance(meta, AttackParam):
                    attack_params[name] = meta
        cls.TEMPLATE_PARAMETERS = template_params
        cls.ATTACK_PARAMETERS = attack_params

    def __init__(self, template: dict, **kwargs):
        for key, meta in self.TEMPLATE_PARAMETERS.items():
            value = template.get(key)
            if value is None:
                value = meta.default
            setattr(self, key, value)

        for key, meta in self.ATTACK_PARAMETERS.items():
            value = kwargs.get(key)
            if value is None:
                value = meta.default
            setattr(self, key, value)

        self.log_file_path = None
        self.setup_logging()
        self.validate_template_inputs()

    def setup_logging(self):
        filename = getattr(self, "log_filename", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_file_path = paths.LOGS / f"{self.attack_type}" / f"{filename}.log"
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

    def _extract(self, combined, features_ds, labels_ds):
        if combined is not None:
            if not isinstance(self.label_column, (int, np.integer)):
                raise ValueError(
                    "A combined dataset requires --label-column. If your labels are in a "
                    "separate file, pass features and labels separately via "
                    "--features-dataset/--labels-array (or --train-features-dataset + "
                    "--train-labels-array + --test-features-dataset + --test-labels-array)."
                )
            X = np.delete(combined.data, self.label_column, axis=1)
            y_raw = combined.data[:, self.label_column]
        else:
            X = features_ds.data
            y_raw = labels_ds.data
        return X, np.asarray(y_raw).ravel()

    def load_train_arrays(self):
        return self._extract(
            self.train_dataset if self.train_dataset is not None else self.dataset,
            self.train_features_dataset if self.train_features_dataset is not None else self.features_dataset,
            self.train_labels_array if self.train_labels_array is not None else self.labels_array,
        )

    def load_test_arrays(self):
        if not self.pre_split:
            return None, None
        return self._extract(self.test_dataset, self.test_features_dataset, self.test_labels_array)