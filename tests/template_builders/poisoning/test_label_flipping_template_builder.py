import json
import pytest

from advplay.attack_templates.template_registry import registry
from advplay.variables import poisoning_techniques, available_attacks
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
        "poisoning_method": poisoning_techniques.LABEL_FLIPPING,
        "framework": "sklearn",
        "algorithm": "logistic_regression",
        "config": fake_training_config_file,
        "source": 0,
        "target": 1,
        "min_portion_to_poison": 0.3,
        "max_portion_to_poison": 0.6,
        "test_portion": 0.2,
        "trigger": None,
        "override": True,
        "filename": "test_template"
    }

@pytest.fixture
def expected_json(label_flipping_template_data):
    with open(label_flipping_template_data["config"], 'r', encoding='utf-8') as config_file:
        training_config = config_file.read()

    return {
        "poisoning_method": label_flipping_template_data["poisoning_method"],
        "training_framework": label_flipping_template_data["framework"],
        "training_algorithm": label_flipping_template_data["algorithm"],
        "training_config": training_config,
        "source_class": label_flipping_template_data["source"],
        "target_class": label_flipping_template_data["target"],
        "min_portion_to_poison": label_flipping_template_data["min_portion_to_poison"],
        "max_portion_to_poison": label_flipping_template_data["max_portion_to_poison"],
        "test_portion": label_flipping_template_data["test_portion"],
        "trigger_pattern": label_flipping_template_data["trigger"],
        "override": label_flipping_template_data["override"]
}

@pytest.fixture
def file_path(label_flipping_template_data: dict, fake_templates_dir) -> str:
    return paths.TEMPLATES / label_flipping_template_data["attack"] / f"{label_flipping_template_data['filename']}.json"

def test_define_template_valid(label_flipping_template_data, file_path, expected_json):
    registry.define_template(
        template_type=label_flipping_template_data["poisoning_method"],
        attack_type=label_flipping_template_data["attack"],
        framework=label_flipping_template_data["framework"],
        algorithm=label_flipping_template_data["algorithm"],
        config=label_flipping_template_data["config"],
        source=label_flipping_template_data["source"],
        target=label_flipping_template_data["target"],
        min_portion_to_poison=label_flipping_template_data["min_portion_to_poison"],
        max_portion_to_poison=label_flipping_template_data["max_portion_to_poison"],
        test_portion=label_flipping_template_data["test_portion"],
        trigger=label_flipping_template_data["trigger"],
        override=label_flipping_template_data["override"],
        filename=label_flipping_template_data["filename"],
    )

    assert file_path.exists(), f"{file_path} was not created"

    with open(file_path) as f:
        data = json.load(f)

    assert  data == expected_json


@pytest.mark.parametrize("bad_kwargs,expected_error", [
    ({"min_portion_to_poison": 1.2}, ValueError),
    ({"max_portion_to_poison": -0.5}, ValueError),
    ({"min_portion_to_poison": 0.8, "max_portion_to_poison": 0.1}, ValueError),
    ({"source": 0, "target": 0}, ValueError),
    ({"override": "yes"}, TypeError),
    ({"filename": ""}, ValueError),
])
def test_define_template_invalid(label_flipping_template_data, bad_kwargs, expected_error):
    kwargs = label_flipping_template_data.copy()
    kwargs.update(bad_kwargs)

    define_kwargs = {
        "template_type": kwargs["poisoning_method"],
        "attack_type": kwargs["attack"],
        "framework": kwargs["framework"],
        "algorithm": kwargs["algorithm"],
        "config": kwargs["config"],
        "source": kwargs["source"],
        "target": kwargs["target"],
        "min_portion_to_poison": kwargs["min_portion_to_poison"],
        "max_portion_to_poison": kwargs["max_portion_to_poison"],
        "test_portion": kwargs["test_portion"],
        "trigger": kwargs["trigger"],
        "override": kwargs["override"],
        "filename": kwargs["filename"],
    }

    with pytest.raises(expected_error):
        registry.define_template(**define_kwargs)
