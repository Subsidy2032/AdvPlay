from abc import ABC, abstractmethod
from datetime import datetime
import json
import os

from advplay import paths

class BaseAttack(ABC):
    registry = {}

    def __init_subclass__(cls, attack_type: str, attack_subtype, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.attack_type = attack_type
        cls.attack_subtype = attack_subtype
        key = (attack_type, attack_subtype)
        BaseAttack.registry[key] = cls

    def __init__(self, template: dict, **kwargs):
        template_params = getattr(super(self.__class__, self), "TEMPLATE_PARAMETERS", {})
        attack_params = getattr(super(self.__class__, self), "ATTACK_PARAMETERS", {})

        for key, meta in template_params.items():
            value = template.get(key)
            if value is None:
                value = meta.get("default")
            setattr(self, key, value)

        for key, meta in attack_params.items():
            value = kwargs.get(key)
            if value is None:
                value = meta.get("default")
            setattr(self, key, value)

        self.log_file_path = None
        self.setup_logging()

    def setup_logging(self):
        filename = getattr(self, "log_filename", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_file_path = paths.ATTACK_LOGS / f"{self.attack_type}" / f"{filename}.log"
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self):
        template_values = {key: getattr(self, key) for key in getattr(self.__class__, "TEMPLATE_PARAMETERS", {})}
        self.save_template(self.template_filename, template_values)

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

    @abstractmethod
    def execute(self):
        raise NotImplementedError("Subclasses must implement the execute method.")
