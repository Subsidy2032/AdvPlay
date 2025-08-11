from openai import OpenAI
from pathlib import Path

from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase
from advplay.variables import available_platforms, default_template_file_names

class OpenAITemplateBuilder(TemplateBuilderBase, template_type=available_platforms.OPENAI):
    def __init__(self, attack_type: str, **kwargs):
        super().__init__(attack_type, **kwargs)
        self.model = self.kwargs.get("model")
        self.instructions = self.kwargs.get("instructions")
        self.filename = self.kwargs.get("filename", default_template_file_names.CUSTOM_INSTRUCTIONS)

    def build(self):
        if self.instructions and Path(self.instructions).exists():
            with open(self.instructions, 'r', encoding='utf-8') as instructions_file:
                self.instructions = instructions_file.read()

        try:
            client = OpenAI()
            models = client.models.list()
            model_names = [model.id for model in models.data]

        except Exception as e:
            print(e)
            model_names = []

        if self.model not in model_names:
            raise NameError(f"An OpenAI model with the name {self.model} does not exist. "
                            f"Some popular OpenAI models are gpt-5 and gpt-5-mini.")

        template = {
            "platform": available_platforms.OPENAI,
            "model": self.model,
            "instructions": self.instructions
        }

        self.save_template(self.filename, template)
