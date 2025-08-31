import json
import pytest

from advplay.variables import poisoning_techniques, available_attacks
from advplay.attacks.attack_runner import define_template
from advplay import paths

@pytest.fixture(autouse=True)
def fake_templates_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("advplay.paths.TEMPLATES", tmp_path)
    return tmp_path

@pytest.fixture
def fake_training_config_file(tmp_path):
    config_file = tmp_path / "fake_training_config.json"
    fake_config = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 5
    }
    config_file.write_text(json.dumps(fake_config))
    return str(config_file)

@pytest.fixture
def label_flipping_template_data(fake_training_config_file, tmp_path):
    return {
        "attack": available_attacks.POISONING,
        "technique": poisoning_techniques.LABEL_FLIPPING,
        "training_framework": "sklearn",
        "training_algorithm": "logistic_regression",
        "training_configuration": fake_training_config_file,
        "test_portion": 0.2,
        "source": 0,
        "target": 1,
        "min_portion_to_poison": 0.3,
        "max_portion_to_poison": 0.6,
        "trigger_pattern": None,
        "override": True,
        "template_filename": "test_template"
    }

@pytest.fixture
def expected_json(label_flipping_template_data):
    with open(label_flipping_template_data["training_configuration"], 'r', encoding='utf-8') as config_file:
        training_config = config_file.read()

    return {
        "technique": label_flipping_template_data["technique"],
        "training_framework": label_flipping_template_data["training_framework"],
        "training_algorithm": label_flipping_template_data["training_algorithm"],
        "training_configuration": label_flipping_template_data["training_configuration"],
        "test_portion": label_flipping_template_data["test_portion"],
        "min_portion_to_poison": label_flipping_template_data["min_portion_to_poison"],
        "max_portion_to_poison": label_flipping_template_data["max_portion_to_poison"],
        "source": label_flipping_template_data["source"],
        "target": label_flipping_template_data["target"],
        "trigger_pattern": label_flipping_template_data["trigger_pattern"],
        "override": label_flipping_template_data["override"]
}

@pytest.fixture
def file_path(label_flipping_template_data: dict, fake_templates_dir) -> str:
    return paths.TEMPLATES / label_flipping_template_data["attack"] / f"{label_flipping_template_data['template_filename']}.json"

def test_define_template_valid(label_flipping_template_data, file_path, expected_json):
    define_template(
        attack_type=label_flipping_template_data["attack"],
        attack_subtype=label_flipping_template_data["technique"],
        training_framework=label_flipping_template_data["training_framework"],
        training_algorithm=label_flipping_template_data["training_algorithm"],
        training_configuration=label_flipping_template_data["training_configuration"],
        test_portion=label_flipping_template_data["test_portion"],
        min_portion_to_poison=label_flipping_template_data["min_portion_to_poison"],
        max_portion_to_poison=label_flipping_template_data["max_portion_to_poison"],
        source=label_flipping_template_data["source"],
        target=label_flipping_template_data["target"],
        trigger_pattern=label_flipping_template_data["trigger_pattern"],
        override=label_flipping_template_data["override"],
        template_filename=label_flipping_template_data["template_filename"],
    )

    assert file_path.exists(), f"{file_path} was not created"

    with open(file_path) as f:
        data = json.load(f)

    assert data == expected_json


@pytest.mark.parametrize("bad_kwargs,expected_error", [
    ({"min_portion_to_poison": 1.2}, ValueError),
    ({"max_portion_to_poison": -0.5}, ValueError),
    ({"min_portion_to_poison": 0.8, "max_portion_to_poison": 0.1}, ValueError),
    ({"source": 0, "target": 0}, ValueError),
    ({"override": "yes"}, TypeError),
    ({"template_filename": ""}, ValueError),
])
def test_define_template_invalid(label_flipping_template_data, bad_kwargs, expected_error):
    kwargs = label_flipping_template_data.copy()
    kwargs.update(bad_kwargs)

    define_kwargs = {
        "attack_type": kwargs["attack"],
        "attack_subtype": kwargs["technique"],
        "training_framework": kwargs["training_framework"],
        "training_algorithm": kwargs["training_algorithm"],
        "training_configuration": kwargs["training_configuration"],
        "test_portion": kwargs["test_portion"],
        "min_portion_to_poison": kwargs["min_portion_to_poison"],
        "max_portion_to_poison": kwargs["max_portion_to_poison"],
        "source": kwargs["source"],
        "target": kwargs["target"],
        "trigger_pattern": kwargs["trigger_pattern"],
        "override": kwargs["override"],
        "template_filename": kwargs["template_filename"],
    }

    with pytest.raises(expected_error):
        define_template(**define_kwargs)
