from abc import ABC
import inspect
from typing import Annotated
import numpy as np
import os
from pathlib import Path

from advplay.attacks.attack_param import AttackParam, TemplateParam
from advplay.attacks.base_attack import BaseAttack
from advplay.variables import default_template_file_names
from advplay.variables import available_attacks
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset
from advplay import paths
from advplay.ml.models.model_loaders.base_model_loader import BaseModelLoader
from advplay.attack_evaluators.contexts.evasion_evaluation_context import EvasionEvaluationContext
from advplay.ml.models.architecture.registry import MODEL_REGISTRY
from advplay.ml.models.loss_functions.registry import LOSS_FUNCTION_REGISTRY

class EvasionAttack(BaseAttack, ABC, attack_type=available_attacks.EVASION, attack_subtype=None):
    training_framework: Annotated[str, BaseAttack.COMMON_TEMPLATE_PARAMETERS['training_framework']]
    model_path: Annotated[str, TemplateParam(type=str, required=True, default=None,
                                             help="Path for the model to load")]
    model: Annotated[str, TemplateParam(type=str, required=True,
                                        help='The training algorithm',
                                        choices=lambda: list({k for k in MODEL_REGISTRY.keys() if k is not None}))]
    training_configuration: Annotated[dict, BaseAttack.COMMON_TEMPLATE_PARAMETERS['training_configuration']]
    preprocessing: Annotated[dict, BaseAttack.COMMON_TEMPLATE_PARAMETERS['preprocessing']]
    model_configuration: Annotated[dict, TemplateParam(type=dict, required=True, default={},
                                                       help="The configurations of the model")]
    data_type: Annotated[str, TemplateParam(type=str, required=False, default="image", help="The data type",
                                            choices=lambda: ["image", "text"])]
    learning_rate: Annotated[float, TemplateParam(type=float, required=False, default=0.01,
                                                  help="The learning rate")]
    template_filename: Annotated[str, TemplateParam(type=str, required=False,
                                                    default=default_template_file_names.EVASION_ATTACK_TEMPLATE,
                                                    help="Template file name")]

    template: Annotated[str, BaseAttack.COMMON_ATTACK_PARAMETERS['template']]
    samples: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=True, default=None,
                                                  help="Samples to run evasions on")]
    confidence: Annotated[float, AttackParam(type=float, required=False, default=0.1,
                                             help="Higher value for more noticeable and robust perturbation")]
    true_labels: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=True, default=None,
                                                      help="The true labels of the provided samples")]
    target_label: Annotated[int, AttackParam(type=int, required=False, default=None,
                                             help="Target labels for misclassification")]
    log_filename: Annotated[str, BaseAttack.COMMON_ATTACK_PARAMETERS['log_filename']]

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

        if self.target_label is not None:
            if "targeted" in inspect.signature(attack_class).parameters:
                attack_instance = attack_class(wrapper, targeted=True, **kwargs)

            else:
                attack_instance = attack_class(wrapper, **kwargs)

            perturbed_samples = attack_instance.generate(x=self.samples_data, y=self.target_label)

        else:
            attack_instance = attack_class(wrapper, **kwargs)
            perturbed_samples = attack_instance.generate(x=self.samples_data)

        context = EvasionEvaluationContext(self.model, None, None, self.training_framework, self.training_configuration, self.model_path, self.samples_data, perturbed_samples, self.target_label)

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
