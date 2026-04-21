from abc import ABC, abstractmethod

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext

class BaseVisualizer(ABC):
    registry = {}

    def __init_subclass__(cls, attack_type: str, attack_subtype: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.attack_type = attack_type
        cls.attack_subtype = attack_subtype
        BaseVisualizer.registry[(attack_type, attack_subtype)] = cls

    @classmethod
    def get(cls, attack_type: str, attack_subtype: str = None):
        visualizer_cls = cls.registry.get((attack_type, attack_subtype))
        if visualizer_cls is None:
            visualizer_cls = cls.registry.get((attack_type, None))
        return visualizer_cls

    @abstractmethod
    def visualize(self, context: BaseVisualizationContext):
        raise NotImplementedError("Subclasses must implement the visualize method.")
