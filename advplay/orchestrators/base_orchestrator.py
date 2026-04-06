from abc import ABC, abstractmethod

class BaseOrchestrator(ABC):
    @abstractmethod
    def run(self, attack_type, attack_subtype, template, command, **kwargs):
        pass
