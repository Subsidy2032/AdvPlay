from datetime import datetime

from advplay.attacks.base_attack import BaseAttack
from advplay.variables import available_attacks, available_platforms
from advplay.attacks.prompt_injection.openai_prompt_injection_attack import OpenAIPromptInjectionAttack

class PromptInjectionAttack(BaseAttack, attack_type=available_attacks.PROMPT_INJECTION):
    def __init__(self, template: dict, **kwargs):
        super().__init__(template, **kwargs)

        self.platform = template.get('platform')
        self.model = template.get('model')
        self.instructions = template.get('instructions')
        self.filename = kwargs.get('filename', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.platforms_cls = {
            available_platforms.OPENAI: OpenAIPromptInjectionAttack
        }

    def execute(self):
        platform_cls = self.platforms_cls.get(self.platform)

        if platform_cls is None:
            raise ValueError(f"Unsupported platform: {self.platform}")

        executor = platform_cls(model=self.model, instructions=self.instructions, filename=self.filename)
        executor.execute()
