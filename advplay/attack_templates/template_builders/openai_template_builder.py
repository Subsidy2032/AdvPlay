from advplay.attack_templates.template_registry.base import TemplateBuilder
from advplay.attack_templates.template_registry.registry import register_template_builder
from advplay.variables import available_platforms
from openai import OpenAI

@register_template_builder(available_platforms.OPENAI)
class OpenAITemplateBuilder(TemplateBuilder):
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