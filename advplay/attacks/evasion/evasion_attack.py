from abc import ABC
import inspect
from typing import Annotated, Union
import numpy as np
import os
from pathlib import Path

from advplay.attacks.attack_param import AttackParam, TemplateParam
from advplay.attacks.base_attack import BaseAttack
from advplay.variables import default_template_file_names, dataset_formats
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
    denormalization: Annotated[dict, BaseAttack.COMMON_TEMPLATE_PARAMETERS['denormalization']]
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
    samples: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=False, default=None,
                                                  help="Samples to run evasions on. Pairs with --true-labels. "
                                                       "Format: '[loader:]path' (prefix overrides extension-based loader lookup).")]
    confidence: Annotated[float, AttackParam(type=float, required=False, default=0.1,
                                             help="Higher value for more noticeable and robust perturbation")]
    true_labels: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=False, default=None,
                                                      help="The true labels of the provided samples. Pairs with --samples. "
                                                           "Format: '[loader:]path' (prefix overrides extension-based loader lookup).")]
    dataset: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=False, default=None,
                                                   help="Combined dataset with samples and labels in one file. Requires --label-column. "
                                                        "Format: '[loader:]path' (prefix overrides extension-based loader lookup).")]
    label_column: Annotated[Union[int, str], BaseAttack.COMMON_ATTACK_PARAMETERS['label_column']]
    target_label: Annotated[int, AttackParam(type=int, required=False, default=None,
                                             help="Target labels for misclassification")]
    log_filename: Annotated[str, BaseAttack.COMMON_ATTACK_PARAMETERS['log_filename']]

    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

        if self.dataset is not None:
            self._unpack_dataset()
        else:
            if self.samples is not None:
                self.samples_data = self.samples.data
            if self.true_labels is not None:
                self.true_labels = self.true_labels.data.ravel()

        if self.target_label is not None:
            self.target_label = np.full(self.true_labels.shape, self.target_label, dtype=self.true_labels.dtype)

    def _unpack_dataset(self):
        if isinstance(self.label_column, str):
            if self.dataset.source_type == dataset_formats.CSV:
                self.label_column = self.dataset.metadata["columns"].get_loc(self.label_column)
            elif self.dataset.source_type == dataset_formats.NPZ:
                key_columns = self.dataset.metadata.get("key_columns", {})
                if self.label_column in key_columns:
                    cols = key_columns[self.label_column]
                    if len(cols) != 1:
                        raise ValueError(
                            f"label_column '{self.label_column}' maps to multiple columns: {cols}"
                        )
                    self.label_column = cols[0]
                else:
                    self.label_column = self.dataset.metadata["keys"].index(self.label_column)
            else:
                raise TypeError("string label_column requires CSV or NPZ format")
        if self.label_column is None:
            raise ValueError("--label-column is required when passing --dataset")

        samples = np.delete(self.dataset.data, self.label_column, axis=1)
        labels = self.dataset.data[:, self.label_column]
        self.samples_data = samples
        self.true_labels = np.asarray(labels).ravel()
        self.samples = LoadedDataset(samples, self.dataset.source_type, self.dataset.metadata)

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

        context = EvasionEvaluationContext(self.model, None, None, self.training_framework, self.training_configuration, self.model_path, self.samples_data, perturbed_samples, self.target_label, true_labels=self.true_labels)

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
