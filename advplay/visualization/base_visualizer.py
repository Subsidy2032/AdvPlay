from abc import ABC, abstractmethod

from advplay.visualization.contexts.base_visualization_context import BaseVisualizationContext

class BaseVisualizer(ABC):
    registry = {}

    def __init_subclass__(cls, attack_type: str):
        BaseVisualizer.registry[attack_type] = cls

    @abstractmethod
    def visualize(self, context: BaseVisualizationContext):
        raise NotImplementedError("Subclasses must implement the visualize method.")