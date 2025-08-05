import json
import os
import pytest
from dotenv import load_dotenv

from advplay.attack_templates.template_registry import registry
from advplay.attack_templates.template_builders.openai_template_builder import OpenAITemplateBuilder
from advplay.paths import LLM_TEMPLATES
from advplay.variables import available_platforms

load_dotenv()

@pytest.fixture
def openai_template_data():
    return {
        "platform": available_platforms.OPENAI,
        "model": "gpt-4o",
        "instructions": "Those are the custom instructions instructions",
        "filename": "new_custom_instructions"
    }

@pytest.fixture
def file_path(openai_template_data):
    return LLM_TEMPLATES / f"{openai_template_data['filename']}.json"

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

    registry.define_template(
        openai_template_data["platform"],
        model=openai_template_data["model"],
        instructions=openai_template_data["instructions"],
        filename=openai_template_data["filename"]
    )

    assert file_path.exists(), f"{file_path} was not created"

    with open(file_path, "r") as f:
        data = json.load(f)

    assert data == expected_json

def test_missing_model(openai_template_data):
    with pytest.raises(TypeError, match="does not exist"):
        registry.define_template(
            openai_template_data["platform"],
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

    registry.define_template(
        openai_template_data["platform"],
        model=openai_template_data["model"],
        filename=openai_template_data["filename"]
    )

    assert file_path.exists()
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == expected_json

def test_default_filename_used(openai_template_data):
    default_file = LLM_TEMPLATES / "custom_instructions.json"
    if default_file.exists():
        os.remove(default_file)

    registry.define_template(
        openai_template_data["platform"],
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
