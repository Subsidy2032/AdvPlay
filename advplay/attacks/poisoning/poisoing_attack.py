from typing import Annotated, Union
import pandas as pd
import numpy as np
from abc import ABC

from advplay.attacks.attack_param import AttackParam, TemplateParam
from advplay.attacks.base_attack import BaseAttack
from advplay.variables import available_attacks, default_template_file_names
from advplay.ml.ops.trainers.base_trainer import BaseTrainer
from advplay.variables import dataset_formats
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset

class PoisoningAttack(BaseAttack, ABC, attack_type=available_attacks.POISONING, attack_subtype=None):
    training_framework: Annotated[str, BaseAttack.COMMON_TEMPLATE_PARAMETERS['training_framework']]
    model: Annotated[str, BaseAttack.COMMON_TEMPLATE_PARAMETERS['model']]
    training_configuration: Annotated[dict, BaseAttack.COMMON_TEMPLATE_PARAMETERS['training_configuration']]
    test_portion: Annotated[float, TemplateParam(type=float, required=True, default=0.2,
                                                 help='Portion of the dataset to be used for testing')]
    min_portion_to_poison: Annotated[float, TemplateParam(type=float, required=True, default=0.1,
                                                          help='Minimum portion of the dataset to poison')]
    max_portion_to_poison: Annotated[float, TemplateParam(type=float, required=False, default=None,
                                                          help='Maximum portion of the dataset to poison')]
    trigger_pattern: Annotated[object, TemplateParam(required=False, default=None,
                                                     help="A trigger to be used for poisoning")]
    override: Annotated[bool, TemplateParam(type=bool, required=False, default=True,
                                            help="Whether to override examples from the training dataset")]
    template_filename: Annotated[str, TemplateParam(type=str, required=False,
                                                    default=default_template_file_names.POISONING_ATTACK_TEMPLATE,
                                                    help="Template file name")]

    template: Annotated[str, BaseAttack.COMMON_ATTACK_PARAMETERS['template']]
    dataset: Annotated[LoadedDataset, BaseAttack.COMMON_ATTACK_PARAMETERS['dataset']]
    features_dataset: Annotated[LoadedDataset, BaseAttack.COMMON_ATTACK_PARAMETERS['features_dataset']]
    labels_array: Annotated[LoadedDataset, BaseAttack.COMMON_ATTACK_PARAMETERS['labels_array']]
    label_column: Annotated[Union[int, str], BaseAttack.COMMON_ATTACK_PARAMETERS['label_column']]
    source: Annotated[Union[int, str], AttackParam(type=(int, str), required=False, default=None,
                                                   help='Source class to poison')]
    target: Annotated[Union[int, str], AttackParam(type=(int, str), required=False, default=None,
                                                   help='Target class')]
    poisoning_dataset: Annotated[pd.DataFrame, AttackParam(type=pd.DataFrame, required=False, default=None,
                                                           help='Poisoned samples to add to the training dataset')]
    seed: Annotated[int, BaseAttack.COMMON_ATTACK_PARAMETERS['seed']]
    step: Annotated[float, AttackParam(type=float, required=False, default=0.1,
                                       help='Incrementing steps to take for poisoning portions')]
    model_name: Annotated[str, BaseAttack.COMMON_ATTACK_PARAMETERS['model_name']]
    log_filename: Annotated[str, BaseAttack.COMMON_ATTACK_PARAMETERS['log_filename']]

    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)
        self.split = False

        if self.dataset is not None:
            self.source_type = self.dataset.source_type
            self.metadata = self.dataset.metadata
            self.dataset_name = self.metadata["dataset_name"]

            if self.source_type == dataset_formats.CSV:
                if isinstance(self.label_column, str):
                    self.label_column = self.metadata["columns"].get_loc(self.label_column)

            elif self.source_type == dataset_formats.NPZ:
                if isinstance(self.label_column, str):
                    key_columns = self.metadata.get("key_columns", {})
                    if self.label_column in key_columns:
                        cols = key_columns[self.label_column]
                        if len(cols) != 1:
                            raise ValueError(
                                f"label_column '{self.label_column}' maps to multiple columns: {cols}"
                            )
                        self.label_column = cols[0]
                    else:
                        self.label_column = self.metadata["keys"].index(self.label_column)

            elif isinstance(self.label_column, str):
                raise TypeError("string column names are only supported for CSV and NPZ formats")

        elif self.features_dataset is not None:
            features_metadata = self.features_dataset.metadata
            labels_metadata = self.labels_array.metadata
            self.features_dataset_name = features_metadata["dataset_name"]
            self.labels_dataset_name = labels_metadata["dataset_name"]

            self.split = True
            self.X_source_type = self.features_dataset.source_type
            self.X_metadata = self.features_dataset.metadata
            self.y_source_type = self.labels_array.source_type
            self.y_metadata = self.labels_array.metadata


    def execute(self):
        self.validate_attack_inputs()

    def validate_attack_inputs(self):
        if self.dataset is not None and type(self.dataset) is not LoadedDataset:
            raise TypeError(f"training_data must be of type LoadedDataset. Got {type(self.dataset)} instead")

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
        if (self.training_framework, self.model) not in BaseTrainer.registry.keys():
            raise ValueError(
                f"Invalid framework and training algorithm configuration: ({self.training_framework}, {self.model})")

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