"""Base class for visualizers, which render an attack's results into plots or tables."""
from abc import ABC, abstractmethod

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext

class BaseVisualizer(ABC):
    """Registry-backed base class for rendering attack results.

    Subclasses register with ``attack_type`` (and optionally ``attack_subtype``); the
    orchestrator selects the technique-specific visualizer first, then falls back to the
    ``(attack_type, None)`` visualizer. Outputs are written under the ``outputs`` directory.

    Attributes:
        registry: Maps ``(attack_type, attack_subtype)`` to the visualizer class.
    """

    registry = {}

    def __init_subclass__(cls, attack_type: str, attack_subtype: str = None, **kwargs):
        """Register the visualizer subclass under ``(attack_type, attack_subtype)``."""
        super().__init_subclass__(**kwargs)
        cls.attack_type = attack_type
        cls.attack_subtype = attack_subtype
        BaseVisualizer.registry[(attack_type, attack_subtype)] = cls

    @classmethod
    def get(cls, attack_type: str, attack_subtype: str = None):
        """Return the visualizer for a technique, falling back to the category default.

        Args:
            attack_type: Attack category to look up.
            attack_subtype: Specific technique; ``None`` selects the category default.

        Returns:
            The matching visualizer class, or ``None`` if none is registered.
        """
        visualizer_cls = cls.registry.get((attack_type, attack_subtype))
        if visualizer_cls is None:
            visualizer_cls = cls.registry.get((attack_type, None))
        return visualizer_cls

    @abstractmethod
    def visualize(self, context: BaseVisualizationContext):
        """Render the attack's results.

        Args:
            context: Visualization context carrying the data to plot or tabulate.
        """
        raise NotImplementedError("Subclasses must implement the visualize method.")
