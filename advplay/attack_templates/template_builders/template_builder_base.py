import json
import os

from advplay import paths

class TemplateBuilderBase:
    registry = {}

    def __init_subclass__(cls, attack_type: str, template_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        TemplateBuilderBase.registry.setdefault(attack_type, {})[template_type] = cls

    def __init__(self, attack_type: str, **kwargs):
        self.attack_type = attack_type
        self.kwargs = kwargs

    def build(self):
        raise NotImplementedError("Subclasses must implement the build method.")

    def save_template(self, filename: str, template: dict):
        template_json = json.dumps(template, indent=4)
        filename = paths.TEMPLATES / self.attack_type / f"{filename}.json"

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
