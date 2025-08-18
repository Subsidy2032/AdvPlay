import json
import pytest
from advplay.attack_templates.template_registry import registry
from advplay.variables import poisoning_techniques


def test_define_template_valid(tmp_path, monkeypatch):
    # Redirect output path
    monkeypatch.setattr("advplay.paths.TEMPLATES", tmp_path)

    attack_type = "valid_attack"
    filename = "valid_file"

    registry.define_template(
        template_type=poisoning_techniques.LABEL_FLIPPING,
        attack_type=attack_type,
        training_framework="sklearn",
        training_algorithm="logistic_regression",
        training_config=None,
        source_class=0,
        target_class=1,
        min_portion_to_poison=0.1,
        max_portion_to_poison=0.3,
        test_portion=0.2,
        trigger_pattern=None,
        override=True,
        filename=filename,
    )

    out_file = tmp_path / attack_type / f"{filename}.json"
    assert out_file.exists()

    with open(out_file) as f:
        data = json.load(f)

    assert data["poisoning_method"] == poisoning_techniques.LABEL_FLIPPING
    assert data["training_framework"] == "sklearn"
    assert data["training_algorithm"] == "logistic_regression"
    assert data["source_class"] == 0
    assert data["target_class"] == 1


@pytest.mark.parametrize("bad_kwargs,expected_error", [
    ({"min_portion_to_poison": 1.2}, ValueError),
    ({"max_portion_to_poison": -0.5}, ValueError),
    ({"min_portion_to_poison": 0.8, "max_portion_to_poison": 0.1}, ValueError),
    ({"source_class": 0, "target_class": 0}, ValueError),
    ({"override": "yes"}, TypeError),
    ({"filename": ""}, ValueError),
])
def test_define_template_invalid(tmp_path, monkeypatch, bad_kwargs, expected_error):
    monkeypatch.setattr("advplay.paths.TEMPLATES", tmp_path)

    kwargs = {
        "template_type": poisoning_techniques.LABEL_FLIPPING,
        "attack_type": "bad_attack",
        "training_framework": "sklearn",
        "training_algorithm": "logistic_regression",
        "source_class": 0,
        "target_class": 1,
        "min_portion_to_poison": 0.1,
        "max_portion_to_poison": 0.2,
        "override": True,
        "filename": "bad",
    }
    kwargs.update(bad_kwargs)

    with pytest.raises(expected_error):
        registry.define_template(**kwargs)


def test_define_template_with_training_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr("advplay.paths.TEMPLATES", tmp_path)

    # fake config file
    config_path = tmp_path / "config.txt"
    config_path.write_text("param=42")

    attack_type = "config_attack"
    filename = "config_template"

    registry.define_template(
        template_type=poisoning_techniques.LABEL_FLIPPING,
        attack_type=attack_type,
        training_framework="sklearn",
        training_algorithm="logistic_regression",
        training_config=str(config_path),
        source_class=1,
        target_class=2,
        min_portion_to_poison=0.1,
        max_portion_to_poison=0.2,
        test_portion=0.25,
        trigger_pattern=None,
        override=True,
        filename=filename,
    )

    out_file = tmp_path / attack_type / f"{filename}.json"
    assert out_file.exists()

    with open(out_file) as f:
        data = json.load(f)

    assert "param=42" in data["training_config"]
