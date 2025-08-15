import json
import os
import pytest
from unittest.mock import patch

from advplay.attacks.attack_runner import attack_runner
from advplay.attacks.prompt_injection.prompt_injection_attack import PromptInjectionAttack
from advplay.variables import available_attacks, available_platforms


@pytest.fixture
def valid_template(tmp_path):
    attack_dir = tmp_path / available_attacks.PROMPT_INJECTION
    attack_dir.mkdir(parents=True)
    template_file = attack_dir / "test_template.json"
    data = {
        "platform": available_platforms.OPENAI,
        "model": "gpt-4o",
        "instructions": "test instructions"
    }
    with open(template_file, "w") as f:
        json.dump(data, f)
    return str(template_file), data


def test_attack_runner_success(valid_template, tmp_path):
    template_path, template_data = valid_template
    with patch("advplay.attacks.attack_runner.TEMPLATES", tmp_path):
        with patch.object(PromptInjectionAttack, "execute") as mock_execute:
            attack_runner(
                attack_type=available_attacks.PROMPT_INJECTION,
                template_name="test_template",
                prompt_list=["test prompt"],
                session_id="sess1"
            )
            mock_execute.assert_called_once()


def test_attack_runner_unsupported_attack(tmp_path):
    with patch("advplay.attacks.attack_runner.TEMPLATES", tmp_path):
        with pytest.raises(ValueError, match="Unsupported attack type"):
            attack_runner("nonexistent_attack", "template")


def test_attack_runner_missing_template(tmp_path):
    attack_dir = tmp_path / available_attacks.PROMPT_INJECTION
    attack_dir.mkdir(parents=True)
    with patch("advplay.attacks.attack_runner.TEMPLATES", tmp_path):
        with pytest.raises(FileNotFoundError, match="file not found"):
            attack_runner(available_attacks.PROMPT_INJECTION, "missing_template")


def test_attack_runner_invalid_json(tmp_path):
    attack_dir = tmp_path / available_attacks.PROMPT_INJECTION
    attack_dir.mkdir(parents=True)
    bad_file = attack_dir / "bad_template.json"
    bad_file.write_text("{invalid json")
    with patch("advplay.attacks.attack_runner.TEMPLATES", tmp_path):
        with pytest.raises(ValueError, match="not a valid json"):
            attack_runner(available_attacks.PROMPT_INJECTION, "bad_template")


def test_prompt_injection_attack_init_and_execute(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("prompt1\nprompt2\n")

    template = {
        "platform": available_platforms.OPENAI,
        "model": "gpt-4o",
        "instructions": "some instructions"
    }

    pia = PromptInjectionAttack(
        template=template,
        prompt_list=[str(prompt_file)],
        session_id="sess123",
        filename="testfile"
    )

    with patch.object(pia, "execute") as mock_execute:
        pia.execute()
        mock_execute.assert_called_once()


def test_prompt_injection_attack_unsupported_platform():
    template = {
        "platform": "unsupported_platform",
        "model": "gpt-4o",
        "instructions": "instructions"
    }
    pia = PromptInjectionAttack(template=template)
    with pytest.raises(ValueError, match="Unsupported platform"):
        pia.execute()
