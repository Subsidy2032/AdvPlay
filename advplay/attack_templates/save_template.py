import advplay.paths as paths

import json
import os
from openai import OpenAI

TEMPLATE_BUILDERS = {}

def register_template_builder(template_type):
    def decorator(func):
        TEMPLATE_BUILDERS[template_type] = func
        return func
    return decorator

@register_template_builder("openai")
def define_openai_template(model, instructions, filename='custom_instructions'):
    model_names = []
    try:
        client = OpenAI()
        models = client.models.list()
        model_names = [model.id for model in models.data]

    except Exception as e:
        print(e)

    if model not in model_names:
        raise TypeError(f"An OpenAI model with the name {model} does not exist.")

    template = {
        "model": model,
        "instructions": instructions
    }

    save_template(filename, template)

def save_template(filename, template):
    template_json = json.dumps(template, indent=4)
    filename = paths.PROMPT_INJECTION_TEMPLATES / f"{filename}.json"

    if filename.exists():
        while True:
            override = input("Configuration file already exists, are you sure you want to replace it? (Y/N) ")
            if override.lower() == 'n':
                return

            elif override.lower() == 'y':
                break

            else:
                print("Invalid option, please choose from the provided options.\n")

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(template_json)

def define_template(template_type, **kwargs):
    builder = TEMPLATE_BUILDERS.get(template_type)
    if builder is None:
        raise ValueError(f"Unsupported template type: {template_type}")

    return builder(**kwargs)