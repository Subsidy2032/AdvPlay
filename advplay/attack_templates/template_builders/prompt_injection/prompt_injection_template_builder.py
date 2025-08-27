from openai import OpenAI
from pathlib import Path

from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase
from advplay.variables import available_platforms, default_template_file_names, available_attacks

class PromptInjectionTemplateBuilder(TemplateBuilderBase, attack_type=available_attacks.PROMPT_INJECTION, template_type=None):
    def __init__(self, attack_type: str, **kwargs):
        super().__init__(attack_type, **kwargs)
        self.model = self.kwargs.get("model")
        self.instructions = self.kwargs.get("instructions")
        self.filename = self.kwargs.get("filename", default_template_file_names.CUSTOM_INSTRUCTIONS)

    def build(self):
        super().build()
