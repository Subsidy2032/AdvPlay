"""Base class for orchestrators that drive an attack run."""
from abc import ABC, abstractmethod

class BaseOrchestrator(ABC):
    """Base class for orchestrators that coordinate a single attack run."""

    @abstractmethod
    def run(self, attack_type, attack_subtype, template, command, **kwargs):
        """Execute an attack identified by ``(attack_type, attack_subtype)`` and template."""
        pass
