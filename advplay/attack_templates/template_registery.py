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


class TemplateBuilder():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self):
        raise NotImplementedError("Subclasses must implement the build method.")

    def save_template(self, filename, template):
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
    builder_cls = TEMPLATE_BUILDERS.get(template_type)
    if builder_cls is None:
        raise ValueError(f"Unsupported template type: {template_type}")

    builder = builder_cls(**kwargs)
    return builder.build()