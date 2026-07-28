"""Base class for orchestrators that drive an attack run."""
from abc import ABC, abstractmethod

class BaseOrchestrator(ABC):
    """Registry-backed base class for orchestrators that coordinate a single attack run.

    Subclasses register by passing ``name`` when they are defined; ``main.py`` looks the name up
    from the ``--orchestrator`` flag and falls back to the default full-pipeline orchestrator when
    none is given. Custom orchestrators (e.g. under ``local/``) opt in the same way.

    Attributes:
        registry: Maps an orchestrator ``name`` to its class.
    """

    registry = {}

    def __init_subclass__(cls, name: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls.name = name
            BaseOrchestrator.registry[name] = cls

    @abstractmethod
    def run(self, attack_type, attack_subtype, template, command, **kwargs):
        """Execute an attack identified by ``(attack_type, attack_subtype)`` and template."""
        pass
