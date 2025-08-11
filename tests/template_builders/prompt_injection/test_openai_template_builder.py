import json
import os
import pytest

from advplay import paths
from advplay.attack_templates.template_registry.registry import define_template
from advplay.paths import TEMPLATES
from advplay.variables import available_platforms, available_attacks, default_template_file_names

@pytest.fixture
def openai_template_data() -> dict:
    return {
        "platform": available_platforms.OPENAI,
        "attack": available_attacks.PROMPT_INJECTION,
        "model": "gpt-4o",
        "instructions": "Those are the custom instructions instructions",
        "filename": "new_custom_instructions"
    }

@pytest.fixture
def file_path(openai_template_data: dict) -> str:
    return TEMPLATES / openai_template_data["attack"] / f"{openai_template_data['filename']}.json"

@pytest.fixture(autouse=True)
def cleanup_file(file_path):
    # Ensure cleanup before and after test
    if file_path.exists():
        os.remove(file_path)
    yield
    if file_path.exists():
        os.remove(file_path)

def test_template_building_with_file_name(openai_template_data, file_path):
    expected_json = {
        "platform": openai_template_data["platform"],
        "model": openai_template_data["model"],
        "instructions": openai_template_data["instructions"]
    }

    define_template(
        openai_template_data["platform"],
        openai_template_data["attack"],
        model=openai_template_data["model"],
        instructions=openai_template_data["instructions"],
        filename=openai_template_data["filename"]
    )

    assert file_path.exists(), f"{file_path} was not created"

    with open(file_path, "r") as f:
        data = json.load(f)

    assert data == expected_json

def test_instructions_from_file(openai_template_data, file_path):
    instruction_file = paths.PROJECT_ROOT / "temp_instructions.txt"
    print(instruction_file)

    with open(instruction_file, "w") as f:
        f.write(openai_template_data["instructions"])

    try:
        define_template(
            openai_template_data["platform"],
            openai_template_data["attack"],
            model=openai_template_data["model"],
            instructions=str(instruction_file),
            filename=openai_template_data["filename"]
        )

        assert file_path.exists(), f"{file_path} was not created"

        with open(file_path, "r") as f:
            data = json.load(f)

        expected_json = {
            "platform": openai_template_data["platform"],
            "model": openai_template_data["model"],
            "instructions": openai_template_data["instructions"]
        }

        assert data == expected_json
    finally:
        if os.path.exists(instruction_file):
            os.remove(instruction_file)

def test_missing_model(openai_template_data):
    with pytest.raises(NameError, match="does not exist"):
        define_template(
            openai_template_data["platform"],
            openai_template_data["attack"],
            model="non-existent-model",
            instructions=openai_template_data["instructions"],
            filename=openai_template_data["filename"]
        )

def test_missing_instructions(openai_template_data, file_path):
    expected_json = {
        "platform": openai_template_data["platform"],
        "model": openai_template_data["model"],
        "instructions": None
    }

    define_template(
        openai_template_data["platform"],
        openai_template_data["attack"],
        model=openai_template_data["model"],
        filename=openai_template_data["filename"]
    )

    assert file_path.exists()
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == expected_json

def test_default_filename_used(openai_template_data):
    default_file = TEMPLATES / openai_template_data["attack"] / f"{default_template_file_names.CUSTOM_INSTRUCTIONS}.json"
    if default_file.exists():
        os.remove(default_file)

    define_template(
        openai_template_data["platform"],
        openai_template_data["attack"],
        model=openai_template_data["model"],
        instructions=openai_template_data["instructions"]
    )

    assert default_file.exists()

    with open(default_file, "r") as f:
        data = json.load(f)

    expected_json = {
        "platform": openai_template_data["platform"],
        "model": openai_template_data["model"],
        "instructions": openai_template_data["instructions"]
    }

    assert data == expected_json

    os.remove(default_file)
