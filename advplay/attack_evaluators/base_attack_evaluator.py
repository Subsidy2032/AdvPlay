from abc import ABC, abstractmethod

from advplay.attack_evaluators.contexts.base_evaluation_context import BaseEvaluationContext

class BaseAttackEvaluator(ABC):
    registry = {}

    def __init_subclass__(cls, attack_type: str, attack_subtype: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.attack_type = attack_type
        cls.attack_subtype = attack_subtype
        BaseAttackEvaluator.registry[(attack_type, attack_subtype)] = cls

    @classmethod
    def get(cls, attack_type: str, attack_subtype: str = None):
        evaluator_cls = cls.registry.get((attack_type, attack_subtype))
        if evaluator_cls is None:
            evaluator_cls = cls.registry.get((attack_type, None))
        return evaluator_cls

    @abstractmethod
    def evaluate(self, context: BaseEvaluationContext):
        raise NotImplementedError("Subclasses must implement the evaluate method.")
