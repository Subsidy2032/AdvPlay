from pathlib import Path

from advplay.attacks.base_attack import BaseAttack
from advplay.variables import available_attacks, available_platforms
from advplay.variables import default_template_file_names
from advplay.model_ops.llms.base_platform import BasePlatform

class PromptInjectionAttack(BaseAttack, attack_type=available_attacks.PROMPT_INJECTION, attack_subtype=None):
    TEMPLATE_PARAMETERS = {
        "platform": {"type": str, "required": True, "default": available_platforms.OPENAI,
                     "help": 'The platform of the LLM', "choices": BasePlatform.registry.keys()},
        "model": {"type": str, "required": True, "default": None, "help": 'The name of the model'},
        "custom_instructions": {"type": str, "required": False, "default": None,
                                "help": 'Custom instructions for the model'},
        "template_filename": {"type": str, "required": False, "default": default_template_file_names.CUSTOM_INSTRUCTIONS,
                              "help": "Template file name"}
    }

    ATTACK_PARAMETERS = {
        "template": BaseAttack.COMMON_ATTACK_PARAMETERS.get('template'),
        "prompt_list": {"type": list, "required": False, "default": None,
                        "help": 'A list or file of prompts to run'},
        "session_id": {"type": str, "required": False, "default": "default_session", "help": 'The session ID'},
        "log_filename": BaseAttack.COMMON_ATTACK_PARAMETERS.get('log_filename')
    }

    def execute(self):
        if self.prompt_list is not None and Path(self.prompt_list[0]).exists():
            with open(self.prompt_list[0], 'r') as prompts_file:
                prompts = [prompt.strip() for prompt in prompts_file]
                self.prompt_list = prompts
        pass

    def build(self):
        if self.custom_instructions and Path(self.custom_instructions).exists():
            with open(self.custom_instructions, 'r', encoding='utf-8') as instructions_file:
                self.custom_instructions = instructions_file.read()

        super().build()
