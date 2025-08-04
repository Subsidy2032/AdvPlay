from advplay.attack_templates.template_registery import *

@register_template_builder("openai")
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
            raise TypeError(f"An OpenAI model with the name {model} does not exist.")

        template = {
            "model": model,
            "instructions": instructions
        }

        self.save_template(filename, template)