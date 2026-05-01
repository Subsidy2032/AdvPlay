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
    preprocessing: Annotated[dict, BaseAttack.COMMON_TEMPLATE_PARAMETERS['preprocessing']]
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
    model_path: Annotated[str, TemplateParam(type=str, required=False, default=None,
                                             help="Path for the model to load")]
    model_configuration: Annotated[dict, TemplateParam(type=dict, required=False, default={},
                                                    help="The configurations of the model")]
    template_filename: Annotated[str, TemplateParam(type=str, required=False,
                                                    default=default_template_file_names.POISONING_ATTACK_TEMPLATE,
                                                    help="Template file name")]

    template: Annotated[str, BaseAttack.COMMON_ATTACK_PARAMETERS['template']]
    dataset: Annotated[LoadedDataset, BaseAttack.COMMON_ATTACK_PARAMETERS['dataset']]
    features_dataset: Annotated[LoadedDataset, BaseAttack.COMMON_ATTACK_PARAMETERS['features_dataset']]
    labels_array: Annotated[LoadedDataset, BaseAttack.COMMON_ATTACK_PARAMETERS['labels_array']]
    train_dataset: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=False, default=None,
                                                        help='Pre-split training dataset (combined with labels). Pairs with --test-dataset.')]
    test_dataset: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=False, default=None,
                                                       help='Pre-split test dataset (combined with labels). Pairs with --train-dataset.')]
    train_features_dataset: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=False, default=None,
                                                                  help='Pre-split training features. Pairs with --train-labels-array and matching test datasets.')]
    train_labels_array: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=False, default=None,
                                                              help='Pre-split training labels. Pairs with --train-features-dataset.')]
    test_features_dataset: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=False, default=None,
                                                                 help='Pre-split test features. Pairs with --test-labels-array.')]
    test_labels_array: Annotated[LoadedDataset, AttackParam(type=LoadedDataset, required=False, default=None,
                                                             help='Pre-split test labels. Pairs with --test-features-dataset.')]
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
        self.pre_split = False

        combined = self.dataset
        if self.train_dataset is not None or self.test_dataset is not None:
            if self.train_dataset is None or self.test_dataset is None:
                raise ValueError("train_dataset and test_dataset must be provided together.")
            self.pre_split = True
            combined = self.train_dataset

        features = self.features_dataset
        labels = self.labels_array
        pre_split_members = (self.train_features_dataset, self.train_labels_array,
                             self.test_features_dataset, self.test_labels_array)
        if any(m is not None for m in pre_split_members):
            if not all(m is not None for m in pre_split_members):
                raise ValueError(
                    "train_features_dataset, train_labels_array, test_features_dataset, and "
                    "test_labels_array must be provided together."
                )
            self.pre_split = True
            features = self.train_features_dataset
            labels = self.train_labels_array

        if combined is not None:
            self.source_type = combined.source_type
            self.metadata = combined.metadata
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

        elif features is not None:
            features_metadata = features.metadata
            labels_metadata = labels.metadata
            self.features_dataset_name = features_metadata["dataset_name"]
            self.labels_dataset_name = labels_metadata["dataset_name"]

            self.split = True
            self.X_source_type = features.source_type
            self.X_metadata = features.metadata
            self.y_source_type = labels.source_type
            self.y_metadata = labels.metadata

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


    def execute(self):
        self.validate_attack_inputs()

    def validate_attack_inputs(self):
        for name, ds in [
            ("dataset", self.dataset),
            ("train_dataset", self.train_dataset),
            ("test_dataset", self.test_dataset),
        ]:
            if ds is not None and type(ds) is not LoadedDataset:
                raise TypeError(f"{name} must be of type LoadedDataset. Got {type(ds)} instead")

        if not self.pre_split and not (0 < self.test_portion < 1):
            raise ValueError("test_portion must be between 0 and 1 when train/test datasets are not supplied separately")

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