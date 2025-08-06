import json
import os

from advplay import paths as paths

class BaseAttack:
    registry = {}

    def __init_subclass__(cls, attack_type: str,  **kwargs):
        super().__init_subclass__(**kwargs)
        BaseAttack.registry[attack_type] = cls

    def __init__(self, template, **kwargs):
        self.template = template
        self.kwargs = kwargs

    def execute(self):
        raise NotImplementedError("Subclasses must implement the execute method.")
