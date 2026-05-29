import json
import pytest

from advplay.variables import available_attacks
from advplay.attacks.base_attack import BaseAttack
from advplay import paths

@pytest.fixture(autouse=True)
def fake_templates_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("advplay.paths.TEMPLATES", tmp_path)
    return tmp_path

@pytest.fixture
def fake_training_config(tmp_path):
    return {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 5
    }

@pytest.fixture
def label_flipping_template_data(fake_training_config, tmp_path):
    return {
        "attack": available_attacks.POISONING,
        "training_framework": "sklearn",
        "model": "logistic_regression",
        "training_configuration": fake_training_config,
        "test_portion": 0.2,
        "min_portion_to_poison": 0.3,
        "max_portion_to_poison": 0.6,
        "trigger_pattern": None,
        "override": True,
        "template_filename": "test_template"
    }

@pytest.fixture
def expected_json(label_flipping_template_data):
    return {
        "training_framework": label_flipping_template_data["training_framework"],
        "model": label_flipping_template_data["model"],
        "training_configuration": label_flipping_template_data["training_configuration"],
        "test_portion": label_flipping_template_data["test_portion"],
        "min_portion_to_poison": label_flipping_template_data["min_portion_to_poison"],
        "max_portion_to_poison": label_flipping_template_data["max_portion_to_poison"],
        "trigger_pattern": label_flipping_template_data["trigger_pattern"],
        "override": label_flipping_template_data["override"],
        'model_configuration': {},
        'model_path': None,
        'preprocessing': None,
        'denormalization': None
}

@pytest.fixture
def file_path(label_flipping_template_data: dict, fake_templates_dir) -> str:
    return paths.TEMPLATES / label_flipping_template_data["attack"] / f"{label_flipping_template_data['template_filename']}.json"

def test_define_template_valid(label_flipping_template_data, file_path, expected_json):
    params = {
        "training_framework": label_flipping_template_data["training_framework"],
        "model": label_flipping_template_data["model"],
        "training_configuration": label_flipping_template_data["training_configuration"],
        "test_portion": label_flipping_template_data["test_portion"],
        "min_portion_to_poison": label_flipping_template_data["min_portion_to_poison"],
        "max_portion_to_poison": label_flipping_template_data["max_portion_to_poison"],
        "trigger_pattern": label_flipping_template_data["trigger_pattern"],
        "override": label_flipping_template_data["override"],
        "template_filename": label_flipping_template_data["template_filename"]
    }
    key = (label_flipping_template_data["attack"], None)
    builder = BaseAttack.registry.get(key)(label_flipping_template_data)
    builder.build()

    assert file_path.exists(), f"{file_path} was not created"

    with open(file_path) as f:
        data = json.load(f)

    assert data == expected_json


@pytest.mark.parametrize("bad_kwargs,expected_error", [
    ({"min_portion_to_poison": 1.2}, ValueError),
    ({"max_portion_to_poison": -0.5}, ValueError),
    ({"min_portion_to_poison": 0.8, "max_portion_to_poison": 0.1}, ValueError),
    ({"override": "yes"}, TypeError),
    ({"template_filename": ""}, ValueError),
])
def test_define_template_invalid(label_flipping_template_data, bad_kwargs, expected_error):
    kwargs = label_flipping_template_data.copy()
    kwargs.update(bad_kwargs)

    params = {
        "attack": kwargs["attack"],
        "training_framework": kwargs["training_framework"],
        "model": kwargs["model"],
        "training_configuration": kwargs["training_configuration"],
        "test_portion": kwargs["test_portion"],
        "min_portion_to_poison": kwargs["min_portion_to_poison"],
        "max_portion_to_poison": kwargs["max_portion_to_poison"],
        "trigger_pattern": kwargs["trigger_pattern"],
        "override": kwargs["override"],
        "template_filename": kwargs["template_filename"]
    }
    key = (kwargs["attack"], None)

    with pytest.raises(expected_error):
        builder = BaseAttack.registry.get(key)(params)
        builder.build()

