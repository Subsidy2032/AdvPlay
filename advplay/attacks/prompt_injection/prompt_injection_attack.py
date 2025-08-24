from datetime import datetime
from pathlib import Path

from advplay.attacks.base_attack import BaseAttack
from advplay.variables import available_attacks, available_platforms
from advplay import paths

class PromptInjectionAttack(BaseAttack, attack_type=available_attacks.PROMPT_INJECTION, attack_subtype=None):
    def __init__(self, template: dict, **kwargs):
        super().__init__(template, **kwargs)
        self.platform = template.get('platform')
        self.model = template.get('model')
        self.instructions = template.get('instructions')

        self.platform = kwargs.get('platform')
        self.prompt_list = kwargs.get('prompt_list', None)
        self.session_id = kwargs.get('session_id', "default_session")
        self.filename = kwargs.get('filename', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.log_file_path = (
                paths.ATTACK_LOGS
                / available_attacks.PROMPT_INJECTION
                / f"{self.filename}.log"
        )
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(self):
        if self.prompt_list and Path(self.prompt_list[0]).exists():
            with open(self.prompt_list[0], 'r') as prompts_file:
                prompts = [prompt.strip() for prompt in prompts_file]
                self.prompt_list = prompts
