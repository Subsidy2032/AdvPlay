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

class EvasionAttack(BaseAttack, ABC, attack_type=available_attacks.EVASION, attack_subtype=None):
    TEMPLATE_PARAMETERS = {
        "technique": BaseAttack.COMMON_TEMPLATE_PARAMETERS.get('technique')(available_attacks.EVASION),
        "training_framework": BaseAttack.COMMON_TEMPLATE_PARAMETERS.get('training_framework'),
        "model_path": {"type": str, "required": True, "default": None, "help": "Path for the model to load"},
        "model_configuration": {"type": dict, "required": True, "default": {},
                                "help": "The configurations of the model"},
        "data_type": {"type": str, "required": False, "default": "image", "help": "The data type",
                      "choices": lambda: ["image", "text"]},
        "learning_rate": {"type": float, "required": False, "default": 0.01, "help": "The learning rate"},
        "eps_step": {"type": float, "required": False, "default": 0.001, "help": "Step size"},
        "max_iter": {"type": int, "required": False, "default": 10, "help": "Maximum iterations"},
        "batch_size": {"type": int, "required": False, "default": 1, "help": "Batch size"},
        "template_filename": {"type": str, "required": False,
                              "default": default_template_file_names.EVASION_ATTACK_TEMPLATE,
                              "help": "Template file name"}
    }

    ATTACK_PARAMETERS = {
        "template": BaseAttack.COMMON_ATTACK_PARAMETERS.get('template'),
        "samples": {"type": LoadedDataset, "required": True, "default": None, "help": "Samples to run evasions on"},
        "eps": {"type": float, "required": False, "default": 0.01, "help": "Maximum perturbation allowed"},
        "theta": {"type": float, "required": False, "default": 0.1,
                  "help": "The amount of perturbation to introduce in each step"},
        "gamma": {"type": float, "required": False, "default": 0.1, "help": "Maximum fraction to effect"},
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

        self.true_labels = self.true_labels.data.ravel()
        self.samples_data = self.samples.data

        if self.target_label is not None:
            self.target_label = np.full_like(self.true_labels.data, self.target_label)

    def execute(self):
        self.validate_attack_inputs()

    def art_evasion(self, attack_class, wrapper, **kwargs):
        if self.target_label is not None:
            if "targeted" in inspect.signature(attack_class).parameters:
                attack_instance = attack_class(wrapper, targeted=True, **kwargs)

            else:
                attack_instance = attack_class(wrapper, **kwargs)

            perturbed_samples = attack_instance.generate(x=self.samples_data, y=self.target_label)

        else:
            attack_instance = attack_class(wrapper, **kwargs)
            perturbed_samples = attack_instance.generate(x=self.samples_data)

        model = registry.load_model(self.training_framework, self.model_path)
        original_predictions = registry.predict(self.training_framework, model, self.samples_data)
        perturbed_predictions = registry.predict(self.training_framework, model, perturbed_samples)
        print(f"Original prediction: {original_predictions}")
        print(f"perturbed image prediction: {perturbed_predictions}")

        num_mispredictions = sum(original != perturbed for original, perturbed in
                                 zip(original_predictions, perturbed_predictions))
        num_samples = len(original_predictions)

        print(f"{(num_mispredictions / num_samples) * 100}% ({num_mispredictions}/{num_samples}) "
              f"of the samples are misclassified after perturbations.")

        if self.target_label is not None:
            num_target_mispredictions = sum(original != 2 and perturbed == 2 for original, perturbed in
                                            zip(original_predictions, perturbed_predictions))
            print(f"{(num_target_mispredictions / num_samples) * 100}% ({num_target_mispredictions}/{num_samples}) "
                  f"of the samples are incorrectly classified as target ({self.target_label}) after perturbations.")
        return perturbed_samples

    def save_perturbed_dataset(self, x_adv):
        dataset_name = self.samples.metadata["dataset_name"]
        dataset_path = paths.DATASETS / 'perturbed_datasets' / f"{dataset_name}_perturbed"
        os.makedirs(dataset_path.parent, exist_ok=True)

        loaded_dataset = LoadedDataset(x_adv, self.samples.source_type, self.samples.metadata)
        registry.save_dataset(loaded_dataset, dataset_path)

        return dataset_path.with_suffix(os.path.splitext(self.samples.metadata["dataset_path"])[1])

    def validate_attack_inputs(self):
        pass

    def validate_template_inputs(self):
        pass