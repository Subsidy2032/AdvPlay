import json
import os
import pytest

from advplay import paths
from advplay.attacks.base_attack import BaseAttack
from advplay.variables import available_platforms, available_attacks, default_template_file_names

@pytest.fixture
def openai_template_data() -> dict:
    return {
        "platform": available_platforms.OPENAI,
        "attack": available_attacks.PROMPT_INJECTION,
        "model": "gpt-4o",
        "custom_instructions": "Those are the custom instructions instructions",
        "template_filename": "test_custom_instructions"
    }

@pytest.fixture(autouse=True)
def fake_templates_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("advplay.paths.TEMPLATES", tmp_path)
    return tmp_path

@pytest.fixture
def file_path(openai_template_data: dict, fake_templates_dir) -> str:
    return paths.TEMPLATES / openai_template_data["attack"] / f"{openai_template_data['template_filename']}.json"

@pytest.fixture
def expected_json(openai_template_data: dict) -> dict:
    return {
        "platform": openai_template_data["platform"],
        "model": openai_template_data["model"],
        "custom_instructions": openai_template_data["custom_instructions"]
    }

@pytest.fixture(autouse=True)
def cleanup_file(file_path):
    if file_path.exists():
        os.remove(file_path)
    yield
    if file_path.exists():
        os.remove(file_path)

def test_template_building_with_file_name(openai_template_data, expected_json, file_path):
    key = (openai_template_data["attack"], None)
    builder = BaseAttack.registry.get(key)(openai_template_data)
    builder.build()

    assert file_path.exists(), f"{file_path} was not created"

    with open(file_path, "r") as f:
        data = json.load(f)

    assert data == expected_json

def test_instructions_from_file(openai_template_data, expected_json, file_path, tmp_path):
    instruction_file = tmp_path / "temp_instructions.txt"
    instruction_file.write_text(openai_template_data["custom_instructions"])

    params = {
        "model": openai_template_data["model"],
        "custom_instructions": str(instruction_file),
        "template_filename": openai_template_data["template_filename"],
    }
    key = (openai_template_data["attack"], None)
    builder = BaseAttack.registry.get(key)(params)
    builder.build()

    assert file_path.exists(), f"{file_path} was not created"

    with open(file_path, "r") as f:
        data = json.load(f)

    assert data == expected_json

def test_default_filename_used(openai_template_data, expected_json):
    default_file = paths.TEMPLATES / openai_template_data["attack"] / f"{default_template_file_names.CUSTOM_INSTRUCTIONS}.json"
    if default_file.exists():
        os.remove(default_file)

    params = {
        "model": openai_template_data["model"],
        "custom_instructions": openai_template_data["custom_instructions"]
    }
    key = (openai_template_data["attack"], None)
    builder = BaseAttack.registry.get(key)(params)
    builder.build()

    assert default_file.exists()

    with open(default_file, "r") as f:
        data = json.load(f)

    assert data == expected_json

def test_missing_instructions(openai_template_data, expected_json, file_path):
    expected_json["custom_instructions"] = None

    params = {
        "attack": openai_template_data["attack"],
        "model": openai_template_data["model"],
        "template_filename": openai_template_data["template_filename"],
    }
    key = (openai_template_data["attack"], None)
    builder = BaseAttack.registry.get(key)(params)
    builder.build()

    assert file_path.exists()
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == expected_json