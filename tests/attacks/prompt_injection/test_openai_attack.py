import json
import os
import pytest
from unittest.mock import patch

from advplay.orchestrators.full_pipeline_orchestrator import FullPipelineOrchestrator
from advplay.loggers.json_logger import JsonLogger
from advplay.attacks.prompt_injection.prompt_injection_attack import PromptInjectionAttack
from advplay.variables import available_attacks


@pytest.fixture
def valid_template(tmp_path):
    return {
        "platform": "openai",
        "model": "gpt-4o",
        "custom_instructions": "test instructions"
    }


@pytest.fixture
def attack_parameters():
    return {
        "attack": available_attacks.PROMPT_INJECTION,
        "technique": "direct",
        "prompt_list": ["test prompt"],
        "session_id": "test",
        "log_filename": "test_log"
    }


@pytest.fixture
def orchestrator(tmp_path, attack_parameters):
    log_path = tmp_path / f"{attack_parameters['log_filename']}.log"
    logger = JsonLogger(log_path)
    return FullPipelineOrchestrator(evaluator=None, logger=logger, visualizer_cls=None)


def run_attack(orchestrator, attack_parameters, valid_template, **overrides):
    kwargs = {
        "prompt_list": attack_parameters["prompt_list"],
        "session_id": attack_parameters["session_id"],
        "log_filename": attack_parameters["log_filename"]
    }
    kwargs.update(overrides)

    orchestrator.run(
        attack_type=attack_parameters['attack'],
        attack_subtype=attack_parameters['technique'],
        template_name=valid_template,
        command="",
        **kwargs
    )


def test_attack_runner_success(orchestrator, attack_parameters, valid_template, tmp_path):
    with patch("advplay.paths.TEMPLATES", tmp_path):
        with patch.object(PromptInjectionAttack, "execute") as mock_execute:
            run_attack(orchestrator, attack_parameters, valid_template)
            mock_execute.assert_called_once()


def test_attack_runner_unsupported_attack(orchestrator, attack_parameters, valid_template, tmp_path):
    with patch("advplay.paths.TEMPLATES", tmp_path):
        with pytest.raises(TypeError, match="'NoneType' object is not callable"):
            orchestrator.run(
                attack_type="nonexistent_attack",
                attack_subtype=attack_parameters['technique'],
                template_name=valid_template,
                command=""
            )


def test_attack_runner_missing_template(orchestrator, attack_parameters, tmp_path):
    attack_dir = tmp_path / attack_parameters["attack"]
    attack_dir.mkdir(parents=True)

    with patch("advplay.paths.TEMPLATES", tmp_path):
        with pytest.raises(FileNotFoundError, match="file not found"):
            orchestrator.run(
                attack_type=attack_parameters["attack"],
                attack_subtype=attack_parameters['technique'],
                template_name="missing_template",
                command=""
            )


def test_attack_runner_invalid_json(orchestrator, attack_parameters, tmp_path):
    attack_dir = tmp_path / attack_parameters["attack"]
    attack_dir.mkdir(parents=True)

    bad_file = attack_dir / "bad_template.json"
    bad_file.write_text("{invalid json")

    with patch("advplay.paths.TEMPLATES", tmp_path):
        with pytest.raises(ValueError, match="not a valid json"):
            orchestrator.run(
                attack_type=attack_parameters["attack"],
                attack_subtype=attack_parameters['technique'],
                template_name="bad_template",
                command=""
            )

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


def test_prompt_injection_attack_unsupported_platform(orchestrator, valid_template, attack_parameters, tmp_path):
    attack_parameters["technique"] = "unsupported_technique"

    with patch("advplay.paths.TEMPLATES", tmp_path):
        with pytest.raises(TypeError, match="'NoneType' object is not callable"):
            run_attack(orchestrator, attack_parameters, valid_template)