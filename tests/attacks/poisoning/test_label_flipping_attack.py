import json
import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
import os

from advplay.attacks.attack_runner import attack_runner
from advplay.variables import available_attacks, poisoning_techniques, available_frameworks, available_models
from advplay import paths
from advplay.model_ops.registry import load_dataset
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset

@pytest.fixture(autouse=True)
def fake_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("advplay.paths.ATTACK_LOGS", tmp_path)
    return tmp_path

@pytest.fixture(autouse=True)
def fake_dataset_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("advplay.paths.DATASETS", tmp_path)

@pytest.fixture(autouse=True)
def fake_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("advplay.paths.MODELS", tmp_path)

@pytest.fixture
def sample_dataset():
    df = pd.DataFrame({
            'feature1': np.arange(10),
            'feature2': np.arange(10, 20),
            'label': [0, 1] * 5
        })
    data = df.to_numpy()

    return LoadedDataset(
        data,
        'csv',
        {"columns": df.columns, "dataset_name": "df", "dataset_path": "datasets/df.csv"}
    )

@pytest.fixture
def single_class_data():
    df = pd.DataFrame({
        'feature1': np.arange(5),
        'feature2': np.arange(5, 10),
        'label': [0] * 5
    })

    data = df.to_numpy()

    return LoadedDataset(
        data,
        'csv',
        {"columns": df.columns, "dataset_name": "df", "dataset_path": "datasets/df.csv"}
    )

@pytest.fixture
def valid_template():
    return {
        "technique": poisoning_techniques.LABEL_FLIPPING,
        "training_framework": available_frameworks.SKLEARN,
        "model": available_models.LOGISTIC_REGRESSION,
        "training_configuration": None,
        "test_portion": 0.2,
        "min_portion_to_poison": 0.3,
        "max_portion_to_poison": 0.4,
        "trigger_pattern": None,
        "override": True
    }

@pytest.fixture
def attack_parameters():
    return {
        "attack": available_attacks.POISONING,
        "technique": poisoning_techniques.LABEL_FLIPPING,
        "seed": 42,
        "label_column": "label",
        "source": 0,
        "target": 1,
        "step": 0.02,
        "model_name": "test_model",
        "log_filename": "test_attack"
    }

@pytest.fixture
def log_file_path(fake_log_dir, attack_parameters):
    return paths.ATTACK_LOGS / attack_parameters["attack"] / f"{attack_parameters['log_filename']}.log"

@pytest.fixture
def dataset_path(fake_dataset_dir, attack_parameters, sample_dataset):
    return (paths.DATASETS / "poisoned_datasets" /
            f"{sample_dataset.metadata['dataset_name']}_poisoned.csv")


@pytest.fixture
def model_path(fake_dataset_dir, attack_parameters, valid_template):
    return (paths.MODELS / valid_template['training_framework'] /
            f"{attack_parameters['model_name']}.joblib")

def test_attack_runs_without_errors(attack_parameters, sample_dataset, valid_template, log_file_path, tmp_path, dataset_path, model_path):
    attack_runner(
        attack_type=attack_parameters['attack'],
        template_name=valid_template,
        dataset=sample_dataset,
        poisoning_data=None,
        seed=attack_parameters["seed"],
        label_column=attack_parameters["label_column"],
        source=attack_parameters["source"],
        target=attack_parameters["target"],
        step=attack_parameters["step"],
        model_name=attack_parameters["model_name"],
        log_filename=attack_parameters["log_filename"]
    )

    assert log_file_path.exists(), f"{log_file_path} was not created"

    try:
        with open(log_file_path, "r") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        assert False, f"{log_file_path} is not valid JSON: {e}"

    assert dataset_path.exists(), f"{dataset_path} was not created"

    ext = os.path.splitext(dataset_path)[1][1:]
    dataset = load_dataset(ext, dataset_path)

    expected = (1 - valid_template['test_portion']) * sample_dataset.data.shape[0]
    assert dataset.data.shape[0] == expected, (
        f"Dataset row count mismatch: expected {expected}, got {dataset.data.shape[0]}"
    )

    assert model_path.exists()

def test_two_log_entries(attack_parameters, sample_dataset, valid_template, log_file_path, dataset_path, model_path):
    attack_runner(
        attack_type=attack_parameters['attack'],
        template_name=valid_template,
        dataset=sample_dataset,
        poisoning_data=None,
        seed=attack_parameters["seed"],
        label_column=attack_parameters["label_column"],
        source=attack_parameters["source"],
        target=attack_parameters["target"],
        step=attack_parameters["step"],
        model_name=attack_parameters["model_name"],
        log_filename=attack_parameters["log_filename"]
    )

    attack_runner(
        attack_type=attack_parameters['attack'],
        template_name=valid_template,
        dataset=sample_dataset,
        poisoning_data=None,
        seed=attack_parameters["seed"],
        label_column=attack_parameters["label_column"],
        source=attack_parameters["source"],
        target=attack_parameters["target"],
        step=attack_parameters["step"],
        model_name=attack_parameters["model_name"],
        log_filename=attack_parameters["log_filename"]
    )

    assert log_file_path.exists(), f"{log_file_path} was not created"

    try:
        with open(log_file_path, "r") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        assert False, f"{log_file_path} is not valid JSON: {e}"

def test_override_false_appends_data(tmp_path, dataset_path, valid_template, attack_parameters, sample_dataset):
    valid_template["override"] = False
    attack_runner(
        attack_type=available_attacks.POISONING,
        template_name=valid_template,
        dataset=sample_dataset,
        poisoning_data=None,
        seed=attack_parameters["seed"],
        label_column=attack_parameters["label_column"],
        source=attack_parameters["source"],
        target=attack_parameters["target"],
        model_name = attack_parameters["model_name"],
        log_filename = attack_parameters["log_filename"]
    )

    assert dataset_path.exists(), f"{dataset_path} was not created"

    ext = os.path.splitext(dataset_path)[1][1:]
    dataset = load_dataset(ext, dataset_path)

    original_num_rows = (1 - valid_template['test_portion']) * sample_dataset.data.shape[0]
    assert dataset.data.shape[0] > original_num_rows, (
        f"Dataset row count mismatch: expected more than {original_num_rows}, got {dataset.data.shape[0]}"
    )

@pytest.mark.parametrize("bad_kwargs,expected_error", [
    ({"label_column": 'fake'}, KeyError),
    ({"step": -0.4}, ValueError),
    ({"seed": "seed"}, TypeError),
])
def test_attack_invalid(attack_parameters, valid_template, bad_kwargs, expected_error, sample_dataset):
    kwargs = attack_parameters.copy()
    kwargs.update(bad_kwargs)

    define_kwargs = {
        "attack_type": kwargs['attack'],
        "template_name": valid_template,
        "dataset": sample_dataset,
        "poisoning_data": None,
        "seed": kwargs["seed"],
        "label_column": kwargs["label_column"],
        "source": kwargs["source"],
        "target": kwargs["target"],
        "step": kwargs["step"],
        "model_name": kwargs["model_name"],
        "log_filename": kwargs["log_filename"]
    }

    with pytest.raises(expected_error):
        attack_runner(**define_kwargs)

def test_invalid_attack_type(valid_template, attack_parameters, tmp_path, sample_dataset):
    with pytest.raises(ValueError, match="Unsupported attack type"):
        attack_runner(
            attack_type="INVALID_ATTACK",
            template_name=valid_template,
            dataset=sample_dataset,
            label_column=attack_parameters["label_column"]
        )

def test_single_class_error(tmp_path, valid_template, attack_parameters, single_class_data):
    with pytest.raises(ValueError, match="Poisoning requires at least two classes"):
        attack_runner(
            attack_type=attack_parameters["attack"],
            template_name=valid_template,
            dataset=single_class_data,
            label_column=attack_parameters["label_column"],
            seed=attack_parameters["seed"]
        )

def test_invalid_template_type(tmp_path, attack_parameters):
    with patch("advplay.utils.load_files.load_json", return_value="not_a_dict"):
        with pytest.raises(TypeError, match="template must be a JSON object"):
            attack_runner(
                attack_type=available_attacks.POISONING,
                template_name="ignored"
            )