# test_attack_runner_extended.py
import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np

from advplay.attacks.attack_runner import attack_runner
from advplay.variables import available_attacks, poisoning_techniques

@pytest.fixture
def sample_training_data():
    return pd.DataFrame({
        'feature1': np.arange(10),
        'feature2': np.arange(10, 20),
        'label': [0, 1] * 5
    })

@pytest.fixture
def single_class_data():
    return pd.DataFrame({
        'feature1': np.arange(5),
        'feature2': np.arange(5, 10),
        'label': [0] * 5
    })

def make_template(**overrides):
    template = {
        "poisoning_method": poisoning_techniques.LABEL_FLIPPING,
        "training_framework": "sklearn",
        "training_algorithm": "logistic_regression",
        "training_config": {},
        "test_portion": 0.2,
        "min_portion_to_poison": 0.1,
        "max_portion_to_poison": 0.2,
        "source_class": 0,
        "target_class": 1,
        "trigger_pattern": None,
        "override": True
    }
    template.update(overrides)
    return template

def test_attack_runs_without_errors(tmp_path, sample_training_data):
    attack_runner(
        attack_type=available_attacks.POISONING,
        template_name=make_template(),
        training_data=sample_training_data,
        poisoning_data=None,
        seed=42,
        label_column='label',
        step=0.05,
        model_name="test_model",
        filename=str(tmp_path / "test_attack")
    )

def test_invalid_attack_type(tmp_path, sample_training_data):
    with pytest.raises(ValueError, match="Unsupported attack type"):
        attack_runner(
            attack_type="INVALID_ATTACK",
            template_name=make_template(),
            training_data=sample_training_data,
            label_column='label'
        )

def test_single_class_error(tmp_path, single_class_data):
    with pytest.raises(ValueError, match="Only one class is present"):
        attack_runner(
            attack_type=available_attacks.POISONING,
            template_name=make_template(),
            training_data=single_class_data,
            label_column='label',
            seed=42
        )

def test_invalid_template_type(tmp_path):
    with patch("advplay.utils.load_files.load_json", return_value="not_a_dict"):
        with pytest.raises(TypeError, match="template must be a JSON object"):
            attack_runner(
                attack_type=available_attacks.POISONING,
                template_name="ignored"
            )

def test_override_false_appends_data(tmp_path, sample_training_data):
    template = make_template(override=False)
    attack_runner(
        attack_type=available_attacks.POISONING,
        template_name=template,
        training_data=sample_training_data,
        poisoning_data=None,
        seed=42,
        label_column='label'
    )

def test_max_less_than_min_error(tmp_path, sample_training_data):
    template = make_template(min_portion_to_poison=0.2, max_portion_to_poison=0.1)
    with pytest.raises(ValueError):
        attack_runner(
            attack_type=available_attacks.POISONING,
            template_name=template,
            training_data=sample_training_data,
            poisoning_data=None,
            label_column='label',
            seed=42
        )
