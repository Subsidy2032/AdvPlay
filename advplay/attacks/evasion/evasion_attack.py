from abc import ABC
import inspect
import numpy as np
import os

from advplay.attacks.base_attack import BaseAttack
from advplay.variables import default_template_file_names
from advplay.variables import available_attacks
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.model_ops import registry
from advplay import paths
from advplay.loggers.json_logger import JsonLogger
from advplay.attack_evaluators.contexts.evasion_context import EvasionContext
from advplay.attack_evaluators.evasion_evaluator import EvasionEvaluator

class EvasionAttack(BaseAttack, ABC, attack_type=available_attacks.EVASION, attack_subtype=None):
    TEMPLATE_PARAMETERS = {
        "training_framework": BaseAttack.COMMON_TEMPLATE_PARAMETERS.get('training_framework'),
        "model_path": {"type": str, "required": True, "default": None, "help": "Path for the model to load"},
        "model_configuration": {"type": dict, "required": True, "default": {},
                                "help": "The configurations of the model"},
        "data_type": {"type": str, "required": False, "default": "image", "help": "The data type",
                      "choices": lambda: ["image", "text"]},
        "learning_rate": {"type": float, "required": False, "default": 0.01, "help": "The learning rate"},
        "template_filename": {"type": str, "required": False,
                              "default": default_template_file_names.EVASION_ATTACK_TEMPLATE,
                              "help": "Template file name"}
    }

    ATTACK_PARAMETERS = {
        "template": BaseAttack.COMMON_ATTACK_PARAMETERS.get('template'),
        "samples": {"type": LoadedDataset, "required": True, "default": None, "help": "Samples to run evasions on"},
        "confidence": {"type": float, "required": False, "default": 0.1,
                       "help": "Higher value for more noticeable and robust perturbation"},
        "true_labels": {"type": LoadedDataset, "required": True, "default": None,
                        "help": "The truelabels of the provided samples"},
        "target_label": {"type": int, "required": False, "default": None,
                         "help": "Target labels for misclassification"},
        "log_filename": BaseAttack.COMMON_ATTACK_PARAMETERS.get('log_filename')
    }

    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

        if self.true_labels:
            self.true_labels = self.true_labels.data.ravel()

        if self.samples:
            self.samples_data = self.samples.data

        if self.target_label is not None:
            self.target_label = np.full(self.true_labels.shape, self.target_label, dtype=self.true_labels.dtype)

    def execute(self):
        self.validate_attack_inputs()

    def art_evasion(self, attack_class, **kwargs):
        wrapper = registry.load_classifier(self.training_framework, self.model_path, self.model_configuration)

        if self.target_label is not None:
            if "targeted" in inspect.signature(attack_class).parameters:
                attack_instance = attack_class(wrapper, targeted=True, **kwargs)

            else:
                attack_instance = attack_class(wrapper, **kwargs)

            perturbed_samples = attack_instance.generate(x=self.samples_data, y=self.target_label)

        else:
            attack_instance = attack_class(wrapper, **kwargs)
            perturbed_samples = attack_instance.generate(x=self.samples_data)

        context = EvasionContext(None, None, None, self.training_framework, None, self.model_path, self.samples_data, perturbed_samples, self.target_label)

        dataset_name = self.samples.metadata["dataset_name"]
        dataset_path = paths.DATASETS / 'perturbed_datasets' / dataset_name
        perturbed_dataset_path = paths.DATASETS / 'perturbed_datasets' / f"{dataset_name}_perturbed"
        os.makedirs(dataset_path.parent, exist_ok=True)

        loaded_dataset = LoadedDataset(perturbed_samples, self.samples.source_type, self.samples.metadata)
        datasets = [(loaded_dataset, str(perturbed_dataset_path))]

        attack_results = {
            "attack": self.attack_type,
            "technique": self.attack_subtype,
            "original_dataset_path": str(dataset_path),
            "perturbed_dataset_path": str(perturbed_dataset_path)
        }
        
        return attack_results, context, datasets

    def validate_attack_inputs(self):
        pass

    def validate_template_inputs(self):
        pass
