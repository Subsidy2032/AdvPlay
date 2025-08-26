from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    registry = {}

    def __init_subclass__(cls, attack_type: str, attack_subtype, **kwargs):
        super().__init_subclass__(**kwargs)
        key = (attack_type, attack_subtype)
        BaseVisualizer.registry[key] = cls

    def __init__(self, log_file, **kwargs):
        self.log_file = log_file
        self.kwargs = kwargs

    @abstractmethod
    def visualize(self):
        raise NotImplementedError("Subclasses must implement the visualize method.")
