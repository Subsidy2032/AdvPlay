import json
import os
import pytest
from unittest.mock import patch

from advplay.attacks.attack_runner import attack_runner
from advplay.attacks.prompt_injection.prompt_injection_attack import PromptInjectionAttack
from advplay.variables import available_attacks, available_platforms


@pytest.fixture
def valid_template(tmp_path):
    return {
        "technique": "openai",
        "model": "gpt-4o",
        "custom_instructions": "test instructions"
    }

@pytest.fixture
def attack_parameters():
    return {
        "attack": available_attacks.PROMPT_INJECTION,
        "technique": available_platforms.OPENAI,
        "prompt_list": ["test prompt"],
        "session_id": "test"
    }

def test_attack_runner_success(attack_parameters, valid_template, tmp_path):
    with patch("advplay.attacks.attack_runner.paths.TEMPLATES", tmp_path):
        with patch.object(PromptInjectionAttack, "execute") as mock_execute:
            attack_runner(
                attack_type=attack_parameters['attack'],
                template_name=valid_template,
                prompt_list=attack_parameters['prompt_list'],
                session_id=attack_parameters['session_id']
            )
            mock_execute.assert_called_once()


def test_attack_runner_unsupported_attack(tmp_path, attack_parameters, valid_template):
    with patch("advplay.attacks.attack_runner.paths.TEMPLATES", tmp_path):
        with pytest.raises(ValueError, match="Unsupported attack type"):
            attack_runner("nonexistent_attack", valid_template)


def test_attack_runner_missing_template(tmp_path, attack_parameters, valid_template):
    attack_dir = tmp_path / attack_parameters["attack"]
    attack_dir.mkdir(parents=True)
    with patch("advplay.attacks.attack_runner.paths.TEMPLATES", tmp_path):
        with pytest.raises(FileNotFoundError, match="file not found"):
            attack_runner(attack_parameters["attack"], "missing_template")


def test_attack_runner_invalid_json(tmp_path, attack_parameters, valid_template):
    attack_dir = tmp_path / attack_parameters["attack"]
    attack_dir.mkdir(parents=True)
    bad_file = attack_dir / "bad_template.json"
    bad_file.write_text("{invalid json")
    with patch("advplay.attacks.attack_runner.paths.TEMPLATES", tmp_path):
        with pytest.raises(ValueError, match="not a valid json"):
            attack_runner(attack_parameters["attack"], "bad_template")


def test_prompt_injection_attack_init_and_execute(tmp_path, attack_parameters, valid_template):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("prompt1\nprompt2\n")

    pia = PromptInjectionAttack(
        template=valid_template,
        prompt_list=[str(prompt_file)],
        session_id=attack_parameters["session_id"],
        log_filename="testfile"
    )

    with patch.object(pia, "execute") as mock_execute:
        pia.execute()
        mock_execute.assert_called_once()


def test_prompt_injection_attack_unsupported_platform(valid_template, attack_parameters):
    valid_template["technique"] = "unsupported_platform"

    with pytest.raises(ValueError, match="Unsupported attack"):
        attack_runner(
            attack_type=attack_parameters['attack'],
            template_name=valid_template,
            prompt_list=attack_parameters['prompt_list'],
            session_id=attack_parameters['session_id']
        )