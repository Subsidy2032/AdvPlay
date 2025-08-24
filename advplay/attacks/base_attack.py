from abc import ABC, abstractmethod

class BaseAttack(ABC):
    registry = {}

    def __init_subclass__(cls, attack_type: str, attack_subtype, **kwargs):
        super().__init_subclass__(**kwargs)
        key = (attack_type, attack_subtype)
        BaseAttack.registry[key] = cls

    def __init__(self, template, **kwargs):
        self.template = template
        self.kwargs = kwargs

    @abstractmethod
    def execute(self):
        raise NotImplementedError("Subclasses must implement the execute method.")
