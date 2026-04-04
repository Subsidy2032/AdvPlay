from abc import ABC, abstractmethod

from advplay.attack_evaluators.contexts.base_evaluation_context import BaseEvaluationContext

class BaseAttackEvaluator(ABC):
    registry = {}

    def __init_subclass__(cls, attack_type: str):
        BaseAttackEvaluator.registry[attack_type] = cls

    @abstractmethod
    def evaluate(self, context: BaseEvaluationContext):
        raise NotImplementedError("Subclasses must implement the evaluate method.")
