import json
import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path

from advplay.variables import available_attacks, poisoning_techniques, available_frameworks, available_models
from advplay import paths
from advplay.model_ops.registry import load_dataset
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.orchestrators.full_pipeline_orchestrator import FullPipelineOrchestrator
from advplay.attack_evaluators.poisoning_evaluator import PoisoningEvaluator
from advplay.loggers.json_logger import JsonLogger


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
    return paths.LOGS / attack_parameters["attack"] / f"{attack_parameters['log_filename']}.log"

@pytest.fixture
def poisoned_dir(fake_dataset_dir):
    return paths.DATASETS / "poisoned_datasets"

@pytest.fixture
def dataset_path(poisoned_dir, sample_dataset):
    return poisoned_dir / f"{sample_dataset.metadata['dataset_name']}_poisoned_"


@pytest.fixture
def model_path(fake_models_dir, attack_parameters, valid_template):
    return (paths.MODELS / valid_template['training_framework'] /
            f"{attack_parameters['model_name']}.joblib")


@pytest.fixture
def orchestrator(log_file_path):
    logger = JsonLogger(log_file_path)
    evaluator = PoisoningEvaluator()
    return FullPipelineOrchestrator(evaluator, logger, visualizer_cls=None)


def run_attack(orchestrator, attack_parameters, valid_template, sample_dataset, **overrides):
    kwargs = {
        "dataset": sample_dataset,
        "poisoning_data": None,
        "seed": attack_parameters["seed"],
        "label_column": attack_parameters["label_column"],
        "source": attack_parameters["source"],
        "target": attack_parameters["target"],
        "step": attack_parameters["step"],
        "model_name": attack_parameters["model_name"],
        "log_filename": attack_parameters["log_filename"]
    }
    kwargs.update(overrides)

    orchestrator.run(
        attack_type=attack_parameters['attack'],
        attack_subtype=attack_parameters['technique'],
        template_name=valid_template,
        command="",
        **kwargs
    )


def test_attack_runs_without_errors(orchestrator, attack_parameters, sample_dataset, valid_template, log_file_path, dataset_path, model_path, poisoned_dir):
    run_attack(orchestrator, attack_parameters, valid_template, sample_dataset)

    assert log_file_path.exists()

    with open(log_file_path, "r") as f:
        json.load(f)

    files = list(poisoned_dir.glob(f"{sample_dataset.metadata['dataset_name']}_poisoned_*"))
    assert len(files) > 0

    file_path = files[0]
    ext = sample_dataset.source_type
    dataset = load_dataset(ext, file_path)

    expected = (1 - valid_template['test_portion']) * sample_dataset.data.shape[0]
    assert dataset.data.shape[0] == expected

    assert model_path.exists()


def test_two_log_entries(orchestrator, attack_parameters, sample_dataset, valid_template, log_file_path):
    run_attack(orchestrator, attack_parameters, valid_template, sample_dataset)
    run_attack(orchestrator, attack_parameters, valid_template, sample_dataset)

    assert log_file_path.exists()

    with open(log_file_path, "r") as f:
        logs = json.load(f)

    assert isinstance(logs, list)
    assert len(logs) == 2


def test_override_false_appends_data(orchestrator, dataset_path, valid_template, attack_parameters, sample_dataset, poisoned_dir):
    valid_template["override"] = False

    run_attack(orchestrator, attack_parameters, valid_template, sample_dataset)

    files = list(poisoned_dir.glob(f"{sample_dataset.metadata['dataset_name']}_poisoned_*"))
    assert len(files) > 0

    file_path = files[0]
    ext = sample_dataset.source_type
    dataset = load_dataset(ext, file_path)

    original_rows = (1 - valid_template['test_portion']) * sample_dataset.data.shape[0]
    assert dataset.data.shape[0] > original_rows


@pytest.mark.parametrize("bad_kwargs,expected_error", [
    ({"label_column": 'fake'}, KeyError),
    ({"step": -0.4}, ValueError),
    ({"seed": "seed"}, TypeError),
])
def test_attack_invalid(orchestrator, attack_parameters, valid_template, bad_kwargs, expected_error, sample_dataset):
    with pytest.raises(expected_error):
        run_attack(orchestrator, attack_parameters, valid_template, sample_dataset, **bad_kwargs)


def test_invalid_attack_type(orchestrator, valid_template, attack_parameters, sample_dataset):
    with pytest.raises(ValueError, match="Unsupported attack type"):
        orchestrator.run(
            attack_type="INVALID_ATTACK",
            attack_subtype=attack_parameters['technique'],
            template_name=valid_template,
            command="",
            dataset=sample_dataset,
            label_column=attack_parameters["label_column"]
        )


def test_single_class_error(orchestrator, valid_template, attack_parameters, single_class_data):
    with pytest.raises(ValueError, match="Poisoning requires at least two classes"):
        run_attack(orchestrator, attack_parameters, valid_template, single_class_data)