from openai import OpenAI

from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase
from advplay.variables import available_platforms

class OpenAITemplateBuilder(TemplateBuilderBase, template_type=available_platforms.OPENAI):
    def build(self):
        model = self.kwargs.get("model")
        instructions = self.kwargs.get("instructions")
        filename = self.kwargs.get("filename", "custom_instructions")

        try:
            client = OpenAI()
            models = client.models.list()
            model_names = [model.id for model in models.data]

        except Exception as e:
            print(e)
            model_names = []

        if model not in model_names:
            raise TypeError(f"An OpenAI model with the name {model} does not exist. "
                            f"Some popular OpenAI models are gpt-4o and gpt-4o-mini.")

        template = {
            "platform": available_platforms.OPENAI,
            "model": model,
            "instructions": instructions
        }

        self.save_template(filename, template)
